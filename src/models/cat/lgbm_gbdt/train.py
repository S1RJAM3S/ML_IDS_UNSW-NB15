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
import framework.base_train as bt

warnings.filterwarnings('ignore')

PATH          = Path(__file__).parent
ARTIFACTS_DIR = PATH / 'artifacts'
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
logger = make_logger(__name__, str(PATH / 'lgbm_gbdt.log'))

N_TRIALS = 10

PARAMS = {
    'boosting_type': 'gbdt',
    'n_estimators': 2000,
    'learning_rate': 0.05,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'class_weight': 'balanced',
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1,
}


class LGBMGBDTTrainer(CatTrainer):
    STAGE = 'cat'

    def __init__(self, artifacts_dir: Path, logger):
        super().__init__(artifacts_dir=artifacts_dir, logger=logger)
        self.num_leaves = -1
        self.min_data_in_leaf = -1
        self.feature_fraction = -1
        self.lambda_l1 = -1
        self.lambda_l2 = -1
        self.n_classes = -1

    def _build_model(self):
        return lgb.LGBMClassifier(
            **PARAMS,
            objective='multiclass',
            num_class=self.n_classes,
            num_leaves=self.num_leaves,
            min_data_in_leaf=self.min_data_in_leaf,
            feature_fraction=self.feature_fraction,
            lambda_l1=self.lambda_l1,
            lambda_l2=self.lambda_l2,
        )

    def _predict_proba(self, model, X: np.ndarray) -> np.ndarray:
        return model.predict_proba(X)

    def _get_smote(self, y_train: np.ndarray):
        return None

    def _fit_model(self, model, X_tr, y_tr, X_val, y_val, sample_weight=None):
        if X_val is not None:
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50, verbose=False),
                    lgb.log_evaluation(period=-1),
                ],
            )
        else:
            model.fit(X_tr, y_tr)
        return model

    def train(self, name: str = 'lgbm_gbdt', label_oof_path: Path = None, label_threshold: float = None):
        X, y_label, y_cat = load_train(stage=self.STAGE)
        X_arr = X.values.astype(np.float64)
        y_arr = y_cat.values.copy()

        if label_oof_path is not None and label_threshold is not None:
            label_oof = np.load(label_oof_path)
            mask  = label_oof >= label_threshold
            X_arr = X_arr[mask]
            y_arr = y_arr[mask]
            n_normal = int((y_arr == 'Normal').sum())
            self.logger.info(f"OOF filter (t={label_threshold}): {mask.sum()} rows kept ({(~mask).sum()} dropped) — {n_normal} Normal FPs retained")

        cnts = Counter(y_arr)
        self.n_classes = len(cnts)
        n_total = sum(cnts.values())
        self.logger.info(f"Train shape: {X_arr.shape}")
        self.logger.info(f"N Classes: {self.n_classes}")
        self.logger.info("Class distribution:")
        for cls, cnt in sorted(cnts.items()):
            min_fold_train = int(cnt * 0.9 * 0.9)  # 90% data, 90% train split in 10-fold
            self.logger.info(f"{cls}: {cnt} - (balanced_weight={n_total/(self.n_classes*cnt)} - min_fold_train~{min_fold_train})")

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def obj(trial):
            self.num_leaves = trial.suggest_int('num_leaves', 63, 255, step=32)
            self.min_data_in_leaf = trial.suggest_int('min_data_in_leaf', 5, 30, step=5)
            self.feature_fraction = trial.suggest_float('feature_fraction', 0.6, 1.0, step=0.1)
            self.lambda_l1 = trial.suggest_float('lambda_l1', 0.0, 0.5, step=0.1)
            self.lambda_l2 = trial.suggest_float('lambda_l2', 0.0, 0.5, step=0.1)

            orig_splits, bt.N_SPLITS = bt.N_SPLITS, 5
            oof_probs, _, classes = self._cv_loop(X_arr, y_arr)
            bt.N_SPLITS = orig_splits

            y_bin  = label_binarize(y_arr, classes=classes)
            pr_auc = average_precision_score(y_bin, oof_probs, average='macro')

            preds = np.array(classes)[oof_probs.argmax(axis=1)]
            self.logger.info(f"Trial {trial.number} - num_leaves={self.num_leaves} min_data={self.min_data_in_leaf} ff={self.feature_fraction} l1={self.lambda_l1} l2={self.lambda_l2} -> Macro PR-AUC={pr_auc}")
            for cls in sorted(classes):
                m = y_arr == cls
                if m.sum() == 0:
                    continue
                r = recall_score(y_arr[m], preds[m], average='micro', zero_division=0)
                self.logger.info(f"{cls}: recall={r} (n={m.sum()})")
            return pr_auc

        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(obj, n_trials=N_TRIALS)

        self.num_leaves = study.best_params['num_leaves']
        self.min_data_in_leaf = study.best_params['min_data_in_leaf']
        self.feature_fraction = study.best_params['feature_fraction']
        self.lambda_l1 = study.best_params['lambda_l1']
        self.lambda_l2 = study.best_params['lambda_l2']
        self.logger.info(f"Best params: {study.best_params} -> Macro PR-AUC={study.best_value}")

        oof_probs, mean_recall, classes = self._cv_loop(X_arr, y_arr)
        self.logger.info(f"Mean OOF Macro Recall = {mean_recall}")

        preds = np.array(classes)[oof_probs.argmax(axis=1)]
        per_cls = {}
        for cls in classes:
            m = y_arr == cls
            if m.sum() == 0:
                continue
            r = recall_score(y_arr[m], preds[m], average='micro', zero_division=0)
            self.logger.info(f"{cls}: recall={r}  (n={m.sum()})")
            per_cls[cls] = float(r)

        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.10, random_state=42)
        tr_idx, val_idx = next(sss.split(X_arr, y_arr))
        probe = self._fit_model(
            self._build_model(),
            X_arr[tr_idx], y_arr[tr_idx],
            X_arr[val_idx], y_arr[val_idx],
        )
        best_iter = probe.best_iteration_
        self.logger.info(f"Early stopping: best_iteration = {best_iter}")

        m_final = lgb.LGBMClassifier(
            **{**PARAMS, 'n_estimators': best_iter},
            objective='multiclass',
            num_class=self.n_classes,
            num_leaves=self.num_leaves,
            min_data_in_leaf=self.min_data_in_leaf,
            feature_fraction=self.feature_fraction,
            lambda_l1=self.lambda_l1,
            lambda_l2=self.lambda_l2,
        )
        m_final.fit(X_arr, y_arr)

        joblib.dump(m_final, ARTIFACTS_DIR / f'{name}.joblib')
        np.save(ARTIFACTS_DIR / f'{name}_oof.npy', oof_probs)

        res = {
            'mean_oof_macro_recall': float(mean_recall),
            'per_class_oof_recall': per_cls,
            'classes': classes,
            'num_leaves': int(self.num_leaves),
            'min_data_in_leaf': int(self.min_data_in_leaf),
            'feature_fraction': float(self.feature_fraction),
            'lambda_l1': float(self.lambda_l1),
            'lambda_l2': float(self.lambda_l2),
            'best_iteration': int(best_iter),
            'n_trials': N_TRIALS,
            'n_splits': 10,
        }
        with open(ARTIFACTS_DIR / f'{name}_res.json', 'w') as f:
            json.dump(res, f)
        self.logger.info(f"RESULT ({name}): {res}")
        return res


if __name__ == '__main__':
    label_path = PATH.parent.parent / 'label' / 'lgbm_fl' / 'artifacts'
    with open(label_path / 'lgbm_fl_res.json', 'r') as f:
        label_threshold = float(json.load(f)['threshold'])
    t = LGBMGBDTTrainer(artifacts_dir=ARTIFACTS_DIR, logger=logger)
    t.train(
        name='lgbm_gbdt',
        label_oof_path = PATH.parent.parent / 'label' / 'lgbm_fl' / 'artifacts' / 'lgbm_fl_oof.npy',
        label_threshold = label_threshold,
    )
