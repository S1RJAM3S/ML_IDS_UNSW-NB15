import warnings
import json
import numpy as np
import joblib
from collections import Counter
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import recall_score
import lightgbm as lgb
import sys
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from utils import make_logger, load_train

warnings.filterwarnings('ignore')

PATH = Path(__file__).parent
ARTIFACTS_DIR = PATH / 'artifacts'
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
CAT_DIR = PATH.parent
logger = make_logger(__name__, str(PATH / 'meta.log'))

BASE_LEARNERS = [
    ('lgbm_gbdt', CAT_DIR / 'lgbm_gbdt'  / 'artifacts' / 'lgbm_gbdt_oof.npy', CAT_DIR / 'lgbm_gbdt'  / 'artifacts' / 'lgbm_gbdt_res.json'),
    ('balanced_rf', CAT_DIR / 'balanced_rf' / 'artifacts' / 'balanced_rf_oof.npy', CAT_DIR / 'balanced_rf' / 'artifacts' / 'balanced_rf_res.json'),
    ('xgboost', CAT_DIR / 'xgboost'    / 'artifacts' / 'xgboost_oof.npy', CAT_DIR / 'xgboost'    / 'artifacts' / 'xgboost_res.json'),
]

META_PARAMS = {
    'objective': 'multiclass',
    'num_class': 10,
    'class_weight': 'balanced',
    'max_depth': 2,
    'num_leaves': 4,
    'min_data_in_leaf': 100,
    'learning_rate': 0.03,
    'n_estimators': 50,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}

N_SPLITS = 10

def load_meta_features(label_oof_path, label_threshold):
    X_cat, _, y_cat = load_train(stage='cat')
    X_arr  = X_cat.values.astype(np.float64)
    y_arr  = y_cat.values

    label_oof = np.load(label_oof_path)
    mask  = label_oof >= label_threshold
    X_arr = X_arr[mask]
    y_arr = y_arr[mask]
    logger.info(f"Stage 1 filter: {mask.sum()} rows kept")

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    _, ho_idx = next(sss.split(X_arr, y_arr))
    X_ho = X_arr[ho_idx]
    y_ho = y_arr[ho_idx]
    logger.info(f"Hold-out for meta-training: {len(X_ho)} samples")

    oofs, classes = [], None
    for name, model_path, res_path in BASE_LEARNERS:
        model = joblib.load(model_path.parent.parent / 'artifacts' / f'{name}.joblib')
        p = model.predict_proba(X_ho)
        oofs.append(p)
        with open(res_path) as f:
            res = json.load(f)
        if classes is None:
            classes = res['classes']
        logger.info(f"{name}: shape={p.shape}")

    X_meta = np.hstack(oofs)
    X_meta = np.clip(X_meta, 1e-7, 1 - 1e-7)
    
    X_meta = np.log(X_meta / (1 - X_meta))
    logger.info(f"Meta-feature matrix: {X_meta.shape}")
    return X_meta, y_ho, classes

def train(label_oof_path: Path, label_threshold: float, name: str = 'meta'):
    X_meta, y_arr, classes = load_meta_features(label_oof_path, label_threshold)
    n_classes = len(classes)

    logger.info(f"Classes: {classes}")
    logger.info(f"Class distribution:")
    for cls, cnt in sorted(Counter(y_arr).items()):
        logger.info(f"{cls}: {cnt}")

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    oof_probs = np.zeros((len(y_arr), n_classes), dtype=np.float64)
    fold_recalls = []

    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    y_arr_idx = np.array([class_to_idx[c] for c in y_arr])

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_meta, y_arr_idx)):
        X_tr, X_val = X_meta[tr_idx], X_meta[val_idx]
        y_tr, y_val = y_arr_idx[tr_idx],  y_arr_idx[val_idx]

        m = lgb.LGBMClassifier(**META_PARAMS)
        m.fit(X_tr, y_tr)

        probs = m.predict_proba(X_val)
        oof_probs[val_idx] = probs

        preds  = np.array(classes)[probs.argmax(axis=1)]
        # Evaluate against true string labels to match prior metrics
        recall = recall_score(y_arr[val_idx], preds, average='macro', zero_division=0)
        fold_recalls.append(recall)
        logger.info(f"Fold {fold+1}/{N_SPLITS} - Macro Recall = {recall}")

    mean_recall = float(np.mean(fold_recalls))
    logger.info(f"Mean OOF Macro Recall: {mean_recall}")

    preds_all = np.array(classes)[oof_probs.argmax(axis=1)]
    per_cls = {}
    logger.info("Per-class OOF Recall:")
    for cls in classes:
        mask = y_arr == cls
        if mask.sum() == 0:
            continue
        r = recall_score(y_arr[mask], preds_all[mask], average='micro', zero_division=0)
        logger.info(f"{cls}: {r}  (n={mask.sum()})")
        per_cls[cls] = float(r)

    # Train final production meta-learner using numerical indices
    meta = lgb.LGBMClassifier(**META_PARAMS)
    meta.fit(X_meta, y_arr_idx)
    
    joblib.dump(meta, ARTIFACTS_DIR / f'{name}.joblib')
    with open(ARTIFACTS_DIR / f'{name}_classes.json', 'w') as f:
        json.dump(classes, f)

    res = {
        'mean_oof_macro_recall': float(mean_recall),
        'per_class_oof_recall': per_cls,
        'classes': classes,
        'n_base_learners': len(BASE_LEARNERS),
        'base_learners': [b[0] for b in BASE_LEARNERS],
        'meta_features': X_meta.shape[1],
        'n_splits': N_SPLITS,
        'meta_params': META_PARAMS,
    }
    with open(ARTIFACTS_DIR / f'{name}_res.json', 'w') as f:
        json.dump(res, f, indent=2)

    logger.info(f"RESULT: mean_oof_macro_recall={mean_recall:.4f}")
    logger.info(f"Per-class: {per_cls}")
    return res

if __name__ == '__main__':
    label_path = PATH.parent.parent / 'label' / 'lgbm_fl' / 'artifacts'
    with open(label_path / 'lgbm_fl_res.json', 'r') as f:
        label_threshold = float(json.load(f)['threshold'])

    train(
        label_oof_path = PATH.parent.parent / 'label' / 'lgbm_fl' / 'artifacts' / 'lgbm_fl_oof.npy',
        label_threshold = label_threshold,
    )
