import warnings
import json
import numpy as np
import joblib
import lightgbm as lgb
import optuna
from collections import Counter
from pathlib import Path
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import recall_score, average_precision_score
import sys
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from framework.base_train import CatTrainer
from utils import make_logger, load_train

warnings.filterwarnings('ignore')
 
PATH          = Path(__file__).parent
ARTIFACTS_DIR = PATH / 'artifacts'
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
logger = make_logger(__name__, str(PATH / 'lgbm_dart.log'))

N_TRIALS = 10

PARAMS = {
    'boosting_type': 'dart',
    'n_estimators': 1000,
    'learning_rate': 0.05,
    'num_leaves': 63,
    'max_depth': -1,
    'min_data_in_leaf': 20,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1,
}

class LGBMDARTTrainer(CatTrainer):
    STAGE = 'cat'

    def __init__(self, artifacts_dir: Path, logger):
        super().__init__(artifacts_dir=artifacts_dir, logger=logger)
        self.drop_rate = -1
        self.skip_drop = -1
        self.n_classes = -1

    def _build_model(self):
        return lgb.LGBMClassifier(**PARAMS, objective='multiclass', num_classes=self.n_classes, drop_rate=self.drop_rate, skip_drop=self.skip_drop)
    
    def _predict_proba(self, model, X: np.ndarray) -> np.ndarray:
        return model.predict_proba(X)

    def _fit_model(self, model, X_tr, y_tr, X_val, y_val, sample_weight=None):
        if X_val is not None:
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False), lgb.log_evaluation(period=-1)])
        else:
            model.fit(X_tr, y_tr)
        return model

    def train(self, name: str = 'lgbm_dart', label_oof_path: Path = None, label_threshold: float = None):
        X, _, y_cat = load_train(stage=self.STAGE)
        X_arr = X.values.astype(np.float64)
        y_arr = y_cat.values
        if label_oof_path is not None and label_threshold is not None:
            label_oof = np.load(label_oof_path)
            mask = label_oof >= label_threshold
            X_arr = X_arr[mask]
            y_arr = y_arr[mask]
            self.logger.info(f"Applied Label filter :>")

        cnts = Counter(y_arr)
        self.n_classes = len(cnts)
        self.logger.info(f"Train shape: {X.shape}")
        self.logger.info(f"N Classes: {self.n_classes}")
        self.logger.info(f"Class distribution:")
        for cls, cnt in sorted(cnts.items()):
            self.logger.info(f"{cls}: {cnt}")

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        def obj(trial):
            self.drop_rate = trial.suggest_float('drop_rate', 0.05, 0.5, step=0.05)
            self.skip_drop = trial.suggest_float('skip_drop', 0.00, 0.50, step=0.05)
            oof_probs, _, classes = self._cv_loop(X_arr, y_arr)
            y_bin = label_binarize(y_arr, classes=classes)
            pr_auc = average_precision_score(y_bin, oof_probs, average='macro')
            self.logger.info(f"Trial {trial.number} - Drop rate = {self.drop_rate} Skip drop = {self.skip_drop} - Macro PR-AUC={pr_auc}")
            return pr_auc

        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(obj, n_trials=N_TRIALS)
        self.drop_rate = study.best_params['drop_rate']
        self.skip_drop = study.best_params['skip_drop']
        self.logger.info(f"Best: drop_rate = {self.drop_rate} - skip_drop = {self.skip_drop} - Macro PR-AUC = {study.best_value}")

        oof_probs, mean_recall, classes = self._cv_loop(X_arr, y_arr)
        self.logger.info(f"Mean OOF Macro Recall = {mean_recall}")
        preds = np.array(classes)[oof_probs.argmax(axis=1)]
        per_cls = {}
        for cls in classes:
            mask = y_arr == cls
            if mask.sum() == 0:
                continue
            r = recall_score(y_arr[mask], preds[mask], average='micro', zero_division=0)
            self.logger.info(f"{cls}: {r} (n={mask.sum()})")
            per_cls[cls] = float(r)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.10, random_state=42)
        tr_idx, val_idx = next(sss.split(X_arr, y_arr))
        probe = self._fit_model(self._build_model(), X_arr[tr_idx], y_arr[tr_idx], X_arr[val_idx], y_arr[val_idx])
        best_iter = probe.best_iteration_
        self.logger.info(f"Early stopping: Best iteration = {best_iter}")

        m = lgb.LGBMClassifier(**{**PARAMS, 'n_estimators': best_iter}, objective='multiclass', num_classes=self.n_classes, drop_rate=self.drop_rate, skip_drop=self.skip_drop)
        m.fit(X_arr, y_arr)

        joblib.dump(m, ARTIFACTS_DIR / f'{name}.joblib')
        np.save(ARTIFACTS_DIR / f'{name}_oof.npy', oof_probs)
        res = {
            'mean_oof_macro_recall': float(mean_recall),
            'per_class_oof_recall': per_cls,
            'classes': classes,
            'drop_rate': float(self.drop_rate),
            'skip_drop': float(self.skip_drop),
            'best_iteration': int(best_iter),
            'n_trials': N_TRIALS,
            'n_splits': 10,
            'smote_target_ratio': self.smote_target_ratio,
            'smote_max_multiplier': self.smote_max_multiplier,
        }
        with open(ARTIFACTS_DIR / f'{name}_res.json', 'w') as f:
            json.dump(res, f)
        self.logger.info(f"RESULT ({name}): {res}")
        return res

if __name__ == '__main__':
    t = LGBMDARTTrainer(artifacts_dir=ARTIFACTS_DIR, logger=logger)
    t.train(label_oof_path=PATH.parent.parent / 'label' / 'lgbm_fl' / 'artifacts' / 'lgbm_fl_oof.npy', label_threshold=0.2513065326633166)
