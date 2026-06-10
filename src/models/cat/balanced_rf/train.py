import warnings
import json
import numpy as np
import joblib
import optuna
from collections import Counter
from pathlib import Path
from sklearn.preprocessing import label_binarize
from sklearn.metrics import recall_score, average_precision_score
from imblearn.ensemble import BalancedRandomForestClassifier
import sys
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from framework.base_train import CatTrainer
from utils import make_logger, load_train

warnings.filterwarnings('ignore')

PATH = Path(__file__).parent
ARTIFACTS_DIR = PATH / 'artifacts'
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
logger = make_logger(__name__, str(PATH / 'balanced_rf.log'))

N_TRIALS = 10


class BalancedRFTrainer(CatTrainer):
    STAGE = 'cat'

    def __init__(self, artifacts_dir: Path, logger):
        super().__init__(artifacts_dir=artifacts_dir, logger=logger)
        self.n_estimators = 300
        self.max_depth = None
        self.max_features = 'sqrt'
        self.min_samples_leaf = 1

    def _build_model(self):
        return BalancedRandomForestClassifier(
            n_estimators = self.n_estimators,
            max_depth = self.max_depth,
            max_features = self.max_features,
            min_samples_leaf = self.min_samples_leaf,
            sampling_strategy = 'all',
            random_state = 42,
            n_jobs = -1
        )

    def _predict_proba(self, model, X: np.ndarray) -> np.ndarray:
        return model.predict_proba(X)   # (n_samples, n_classes)

    def _get_smote(self, y_train):
        return None

    def _fit_model(self, model, X_tr, y_tr, X_val, y_val, sample_weight=None):
        model.fit(X_tr, y_tr)
        return model

    def train(self, name: str = 'balanced_rf', label_oof_path: Path = None, label_threshold: float = None):
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

        cnts = Counter(y_arr)
        self.logger.info(f"Train shape: {X.shape}")
        self.logger.info(f"N Classes: {len(cnts)}")
        self.logger.info(f"Class distribution:")
        n_total = sum(cnts.values())
        for cls, cnt in sorted(cnts.items()):
            min_fold_train = int(cnt * 0.9 * 0.9)
            self.logger.info(f"{cls}: {cnt} - (balanced_weight={n_total/(len(cnts)*cnt)} - min_fold_train~{min_fold_train})")

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def obj(trial):
            self.max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.3, 0.5])
            self.min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
            self.max_depth = trial.suggest_categorical('max_depth', [None, 10, 20, 30])

            oof_probs, _, classes = self._cv_loop(X_arr, y_arr)
            y_bin  = label_binarize(y_arr, classes=classes)
            pr_auc = average_precision_score(y_bin, oof_probs, average='macro')

            preds = np.array(classes)[oof_probs.argmax(axis=1)]
            self.logger.info(f"Trial {trial.number} - max_features={self.max_features} - min_samples_leaf={self.min_samples_leaf} - max_depth={self.max_depth} -> Macro PR-AUC={pr_auc}")
            for cls in sorted(classes):
                m = y_arr == cls
                if m.sum() == 0:
                    continue
                r = recall_score(y_arr[m], preds[m], average='micro', zero_division=0)
                self.logger.info(f"{cls}: recall={r} (n={m.sum()})")
            return pr_auc

        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        study.optimize(obj, n_trials=N_TRIALS)
        self.max_features = study.best_params['max_features']
        self.min_samples_leaf = study.best_params['min_samples_leaf']
        self.max_depth = study.best_params['max_depth']
        self.logger.info(f"Best params: {study.best_params} -> Macro PR-AUC={study.best_value}")

        oof_probs, mean_recall, classes = self._cv_loop(X_arr, y_arr)
        self.logger.info(f"Mean OOF Macro Recall = {mean_recall}")

        preds = np.array(classes)[oof_probs.argmax(axis=1)]
        per_cls = {}
        for cls in classes:
            mask = y_arr == cls
            if mask.sum() == 0:
                continue
            r = recall_score(y_arr[mask], preds[mask], average='micro', zero_division=0)
            self.logger.info(f"{cls}: recall={r}  (n={mask.sum()})")
            per_cls[cls] = float(r)

        self.n_estimators = 500
        m = self._build_model()
        m.fit(X_arr, y_arr)

        joblib.dump(m, ARTIFACTS_DIR / f'{name}.joblib')
        np.save(ARTIFACTS_DIR / f'{name}_oof.npy', oof_probs)

        res = {
            'mean_oof_macro_recall': float(mean_recall),
            'per_class_oof_recall': per_cls,
            'classes': classes,
            'max_features': str(self.max_features),
            'min_samples_leaf': int(self.min_samples_leaf),
            'max_depth': self.max_depth,
            'n_estimators_cv': 300,
            'n_estimators_final': 500,
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
    label_path = PATH.parent.parent / 'label' / 'lgbm_fl' / 'artifacts'
    with open(label_path / 'lgbm_fl_res.json', 'r') as f:
        label_threshold = float(json.load(f)['threshold'])
    t = BalancedRFTrainer(artifacts_dir=ARTIFACTS_DIR, logger=logger)
    t.train(
        label_oof_path = label_path / 'lgbm_fl_oof.npy',
        label_threshold = label_threshold,
    )
