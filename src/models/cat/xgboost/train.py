import warnings
import json
import numpy as np
import joblib
import optuna
from collections import Counter
from pathlib import Path
from sklearn.preprocessing import label_binarize, LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import recall_score, average_precision_score
from xgboost import XGBClassifier
import sys
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from framework.base_train import CatTrainer
from utils import make_logger, load_train

warnings.filterwarnings('ignore')

PATH = Path(__file__).parent
ARTIFACTS_DIR = PATH / 'artifacts'
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
logger = make_logger(__name__, str(PATH / 'xgboost.log'))

N_TRIALS = 10

PARAMS = {
    'n_estimators': 4000,
    'learning_rate': 0.05,
    'objective': 'multi:softprob',
    'tree_method': 'hist',
    'random_state': 42,
    'n_jobs': -1,
    'verbosity': 0,
}


class XGBoostTrainer(CatTrainer):
    STAGE = 'cat'

    def __init__(self, artifacts_dir: Path, logger):
        super().__init__(artifacts_dir=artifacts_dir, logger=logger)
        self.max_depth = 6
        self.subsample = 0.8
        self.colsample_bytree = 0.8
        self.min_child_weight = 1
        self.reg_lambda = 1.0
        self.reg_alpha = 0.0
        self.n_classes = -1

    def _build_model(self, early_stopping_rounds=None):
        return XGBClassifier(
            **PARAMS,
            num_class = self.n_classes,
            max_depth = self.max_depth,
            subsample = self.subsample,
            colsample_bytree = self.colsample_bytree,
            min_child_weight = self.min_child_weight,
            reg_lambda = self.reg_lambda,
            reg_alpha = self.reg_alpha,
            early_stopping_rounds = early_stopping_rounds,
        )

    def _predict_proba(self, model, X: np.ndarray) -> np.ndarray:
        return model.predict_proba(X)

    def _get_smote(self, y_train):
        return None

    def _fit_model(self, model, X_tr, y_tr, X_val, y_val, sample_weight=None):
        sw = compute_sample_weight('balanced', y_tr)
        if X_val is not None:
            fit_kwargs = {
                "sample_weight": sw,
                "eval_set": [(X_val, y_val)],
                "verbose": False
            }
            model.fit(X_tr, y_tr, **fit_kwargs)
        else:
            model.fit(X_tr, y_tr, sample_weight=sw)
        return model

    def train(self, name: str = 'xgboost', label_oof_path: Path = None, label_threshold: float = None):
        X, _, y_cat = load_train(stage=self.STAGE)
        X_arr = X.values.astype(np.float64)
        y_arr = y_cat.values

        if label_oof_path is not None and label_threshold is not None:
            label_oof = np.load(label_oof_path)
            mask = label_oof >= label_threshold
            X_arr = X_arr[mask]
            y_arr = y_arr[mask]
            n_normal = int((y_arr == 'Normal').sum())
            self.logger.info(f"OOF filter (t={label_threshold}): {mask.sum()} rows kept ({(~mask).sum()} dropped) — {n_normal} Normal FPs retained")

        le = LabelEncoder()
        y_encoded = le.fit_transform(y_arr)
        string_classes = list(le.classes_)  # Order matches integer values [0, 1, 2...]

        cnts = Counter(y_arr)
        self.n_classes = len(cnts)
        self.logger.info(f"Train shape: {X.shape}")
        self.logger.info(f"N Classes: {self.n_classes}")
        self.logger.info(f"Class distribution:")
        n_total = sum(cnts.values())
        for cls, cnt in sorted(cnts.items()):
            min_fold_train = int(cnt * 0.9 * 0.9)
            self.logger.info(f"{cls}: {cnt} - (balanced_weight={n_total/(self.n_classes*cnt):.1f} - min_fold_train~{min_fold_train})")

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def obj(trial):
            self.max_depth = trial.suggest_int('max_depth', 3, 8)
            self.subsample = trial.suggest_float('subsample', 0.6, 1.0, step=0.1)
            self.colsample_bytree = trial.suggest_float('colsample_bytree', 0.6, 1.0, step=0.1)
            self.min_child_weight = trial.suggest_int('min_child_weight', 1, 10)
            self.reg_lambda = trial.suggest_float('reg_lambda', 0.1, 5.0, log=True)
            self.reg_alpha = trial.suggest_float('reg_alpha', 0.0, 2.0, step=0.1)

            oof_probs, _, classes = self._cv_loop(X_arr, y_encoded)
            
            # NOTE: Since y_encoded is integers [0...N-1], classes returned from _cv_loop will also be [0...N-1]
            y_bin  = label_binarize(y_encoded, classes=classes)
            pr_auc = average_precision_score(y_bin, oof_probs, average='macro')

            pred_ints = oof_probs.argmax(axis=1)
            preds = le.inverse_transform(pred_ints)
            
            self.logger.info(f"Trial {trial.number} - max_depth={self.max_depth} subsample={self.subsample:.1f} colsample={self.colsample_bytree:.1f} min_child_w={self.min_child_weight} l2={self.reg_lambda:.2f} l1={self.reg_alpha:.1f} -> Macro PR-AUC={pr_auc:.6f}")
            for cls in sorted(string_classes):
                m = y_arr == cls
                if m.sum() == 0:
                    continue
                r = recall_score(y_arr[m], preds[m], average='micro', zero_division=0)
                self.logger.info(f"{cls}: recall={r:.4f} (n={m.sum()})")
            return pr_auc

        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(obj, n_trials=N_TRIALS)
        self.max_depth = study.best_params['max_depth']
        self.subsample = study.best_params['subsample']
        self.colsample_bytree = study.best_params['colsample_bytree']
        self.min_child_weight = study.best_params['min_child_weight']
        self.reg_lambda = study.best_params['reg_lambda']
        self.reg_alpha = study.best_params['reg_alpha']
        self.logger.info(f"Best params: {study.best_params} -> Macro PR-AUC={study.best_value:.6f}")

        oof_probs, mean_recall, classes = self._cv_loop(X_arr, y_encoded)
        self.logger.info(f"Mean OOF Macro Recall = {mean_recall:.4f}")

        pred_ints = oof_probs.argmax(axis=1)
        preds = le.inverse_transform(pred_ints)
        
        per_cls = {}
        for cls in string_classes:
            mask = y_arr == cls
            if mask.sum() == 0:
                continue
            r = recall_score(y_arr[mask], preds[mask], average='micro', zero_division=0)
            self.logger.info(f"{cls}: recall={r:.4f}  (n={mask.sum()})")
            per_cls[cls] = float(r)

        self.logger.info("Finding best_iteration via 10% hold-out...")
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.10, random_state=42)
        tr_idx, val_idx = next(sss.split(X_arr, y_encoded))
        
        early_stopping_model = self._build_model(early_stopping_rounds=50)
        
        probe = self._fit_model(
            early_stopping_model,
            X_arr[tr_idx], y_encoded[tr_idx],
            X_arr[val_idx], y_encoded[val_idx]
        )
        best_iter = probe.best_iteration
        if best_iter is None or best_iter <= 0:
            best_iter = PARAMS['n_estimators']
            self.logger.info(f"Early stopping did not trigger — using n_estimators={best_iter}")
        else:
            self.logger.info(f"Early stopping: best_iteration = {best_iter}")

        m = XGBClassifier(
            **{**PARAMS, 'n_estimators': best_iter},
            num_class = self.n_classes,
            max_depth = self.max_depth,
            subsample = self.subsample,
            colsample_bytree = self.colsample_bytree,
            min_child_weight = self.min_child_weight,
            reg_lambda = self.reg_lambda,
            reg_alpha = self.reg_alpha,
        )
        sw_full = compute_sample_weight('balanced', y_encoded)
        m.fit(X_arr, y_encoded, sample_weight=sw_full)
        self.logger.info("Final model trained on full dataset.")

        joblib.dump(m, ARTIFACTS_DIR / f'{name}.joblib')
        joblib.dump(le, ARTIFACTS_DIR / f'{name}_encoder.joblib')
        np.save(ARTIFACTS_DIR / f'{name}_oof.npy', oof_probs)

        res = {
            'mean_oof_macro_recall': float(mean_recall),
            'per_class_oof_recall': per_cls,
            'classes': string_classes, 
            'max_depth': int(self.max_depth),
            'subsample': float(self.subsample),
            'colsample_bytree': float(self.colsample_bytree),
            'min_child_weight': int(self.min_child_weight),
            'reg_lambda': float(self.reg_lambda),
            'reg_alpha': float(self.reg_alpha),
            'best_iteration': int(best_iter),
            'n_trials': N_TRIALS,
            'n_splits': 10
        }
        with open(ARTIFACTS_DIR / f'{name}_res.json', 'w') as f:
            json.dump(res, f)
        self.logger.info(f"RESULT ({name}): {res}")
        return res


if __name__ == '__main__':
    t = XGBoostTrainer(artifacts_dir=ARTIFACTS_DIR, logger=logger)
    t.train(
        label_oof_path  = PATH.parent.parent / 'label' / 'lgbm_fl' / 'artifacts' / 'lgbm_fl_oof.npy',
        label_threshold = 0.2513065326633166,
    )
