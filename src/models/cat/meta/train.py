import warnings
import json
import numpy as np
import joblib
from collections import Counter
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import recall_score
from sklearn.linear_model import LogisticRegression
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
    'class_weight': 'balanced',
    'penalty': 'l2',
    'C': 2.0, # Inverse regularization strength
    'max_iter': 1000,
    'random_state': 42,
    'n_jobs': -1
}

N_SPLITS = 10


def load_meta_features(label_oof_path: Path, label_threshold: float) -> tuple:
    _, _, y_cat = load_train(stage='cat')
    y_arr_full  = y_cat.values
    label_oof = np.load(label_oof_path)
    mask      = label_oof >= label_threshold
    y_arr     = y_arr_full[mask]
    logger.info(f"Stage 1 filter: {mask.sum()} rows kept ({(~mask).sum()} dropped)")
    oofs  = []
    classes = None
    for name, oof_path, res_path in BASE_LEARNERS:
        assert oof_path.exists(), f"{name} OOF not found at {oof_path} — run base learner first"
        oof = np.load(oof_path)
        assert oof.shape[0] == y_arr.shape[0], (f"{name} OOF shape {oof.shape} does not match y shape {y_arr.shape} - Ensure all base learners used the same Stage 1 filter.")
        oofs.append(oof)
        with open(res_path) as f:
            res = json.load(f)
        if classes is None:
            classes = res['classes']

    X_meta = np.hstack(oofs)
    
    X_meta = np.clip(X_meta, 1e-7, 1 - 1e-7)
    X_meta = np.log(X_meta / (1 - X_meta))

    logger.info(f"Meta-feature matrix: {X_meta.shape}  ({len(BASE_LEARNERS)} models × {oofs[0].shape[1]} classes)")
    return X_meta, y_arr, classes


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

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_meta, y_arr)):
        X_tr, X_val = X_meta[tr_idx], X_meta[val_idx]
        y_tr, y_val = y_arr[tr_idx],  y_arr[val_idx]

        m = LogisticRegression(**META_PARAMS)
        m.fit(X_tr, y_tr)

        probs = m.predict_proba(X_val)
        oof_probs[val_idx] = probs

        preds  = np.array(classes)[probs.argmax(axis=1)]
        recall = recall_score(y_val, preds, average='macro', zero_division=0)
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

    meta = LogisticRegression(**META_PARAMS)
    meta.fit(X_meta, y_arr)
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
    train(
        label_oof_path = PATH.parent.parent / 'label' / 'lgbm_fl' / 'artifacts' / 'lgbm_fl_oof.npy',
        label_threshold = 0.2513065326633166,
    )
