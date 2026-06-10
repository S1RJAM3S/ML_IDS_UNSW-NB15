import warnings
import json
import numpy as np
import joblib
import lightgbm as lgb
import optuna
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import recall_score, precision_score, matthews_corrcoef, average_precision_score
import sys
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from framework.base_train import LabelTrainer
from utils import make_logger, load_train

warnings.filterwarnings('ignore')

PATH = Path(__file__).parent
ARTIFACTS_DIR = PATH / 'artifacts'
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

logger = make_logger(__name__, str(PATH / 'lgbm_fl.log'))

N_TRIALS = 30


class FLLGBM:
    def __init__(self, gamma: float, alpha: float):
        self.gamma = gamma
        self.alpha = alpha

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray):
        y = y_true.astype(int)
        p = 1.0 / (1.0 + np.exp(-y_pred))
        p_t = np.where(y == 1, p, 1.0 - p)
        alpha_t = np.where(y == 1, self.alpha, 1.0 - self.alpha)
        grad = alpha_t * ((1 - p_t) ** self.gamma) * (self.gamma * p_t * np.log(p_t + 1e-9) + p_t - 1)
        grad = np.where(y == 1, grad, -grad)
        hess = alpha_t * (1.0 - p_t) ** self.gamma * p * (1.0 - p)
        return grad, hess


class FLLGBMEvalError:
    def __init__(self, gamma: float):
        self.gamma = gamma

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray):
        y = y_true.astype(int)
        p = 1.0 / (1.0 + np.exp(-y_pred))
        p_t = np.where(y == 1, p, 1.0 - p)
        loss = -((1 - p_t) ** self.gamma) * np.log(p_t + 1e-9)
        return 'focal_loss', float(loss.mean()), False  # False = lower is better :>


class FLLGBMWrapper(lgb.LGBMClassifier):
    def predict_proba(self, X, **kwargs):
        raw = self.booster_.predict(X)
        p = 1.0 / (1.0 + np.exp(-raw))
        return np.column_stack([1.0 - p, p])


PARAMS = {
    'n_estimators': 2000,
    'learning_rate': 0.05,
    'num_leaves': 48,
    'max_depth': -1,
    'min_data_in_leaf': 50,
    'bagging_fraction': 0.7,
    'bagging_freq': 1,
    'feature_fraction': 0.7,
    'lambda_l1': 0.5,
    'lambda_l2': 1.0,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1,
}


class FLLGBMTrainer(LabelTrainer):
    STAGE = 'label'  # loads selected_label_cols from artifacts

    def __init__(self, artifacts_dir: Path, logger):
        super().__init__(artifacts_dir=artifacts_dir, logger=logger)
        self.gamma = -1
        self.alpha = -1

    def _build_model(self):
        return FLLGBMWrapper(**PARAMS, objective=FLLGBM(self.gamma, self.alpha))

    def _predict_proba(self, model, X: np.ndarray) -> np.ndarray:
        return model.predict_proba(X)[:, 1]

    def _fit_model(self, model, X_tr, y_tr, X_val, y_val, sample_weight=None):
        if X_val is not None:
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                eval_metric=FLLGBMEvalError(self.gamma),
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50, verbose=False),
                    lgb.log_evaluation(period=-1),
                ],
            )
        else:
            model.fit(X_tr, y_tr)
        return model

    def train(self, name: str = 'lgbm_fl'):
        X, y_label, _ = load_train(stage=self.STAGE)
        X_arr = X.values.astype(np.float64)
        y_arr = y_label.values.astype(int)
        attack_rate = float(y_arr.mean())
        self.logger.info(f"Train shape: {X.shape}")
        self.logger.info(f"Attack rate: {attack_rate}")

        alpha_mid = 1.0 - attack_rate
        alpha_low = max(0.20, alpha_mid - 0.10)
        alpha_high = min(0.55, alpha_mid + 0.15)
        self.logger.info(f"Alpha search range: [{alpha_low}, {alpha_high}]")
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def obj(trial):
            gamma = trial.suggest_float('gamma', 1.0, 4.0, step=0.25)
            alpha = trial.suggest_float('alpha', alpha_low, alpha_high, step=0.05)
            self.gamma = gamma
            self.alpha = alpha
            oof_probs, _ = self._cv_loop(X_arr, y_arr)
            pr_auc = average_precision_score(y_arr, oof_probs)
            self.logger.info(f"Trial {trial.number} - gamma={gamma}, alpha={alpha} - OOF PR-AUC={pr_auc}")
            return pr_auc

        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(obj, n_trials=N_TRIALS)
        self.gamma = study.best_params['gamma']
        self.alpha = study.best_params['alpha']
        self.logger.info(f"Best: gamma={self.gamma}, alpha={self.alpha} - OOF PR-AUC={study.best_value}")

        oof_probs, mean_recall = self._cv_loop(X_arr, y_arr)
        self.logger.info(f"Mean OOF Recall@0.5 = {mean_recall}")
        np.save(ARTIFACTS_DIR / 'lgbm_fl_oof.npy', oof_probs)
        best_t, best_f2 = self._find_threshold(y_arr, oof_probs)
        best_pred = (oof_probs >= best_t).astype(int)

        r = recall_score(y_arr, best_pred, zero_division=0)
        p = precision_score(y_arr, best_pred, zero_division=0)
        mcc = matthews_corrcoef(y_arr, best_pred)
        self.logger.info(f"Best threshold = {best_t}")
        self.logger.info(f"OOF Recall = {r}")
        self.logger.info(f"OOF Precision = {p}")
        self.logger.info(f"OOF F2 = {best_f2}")
        self.logger.info(f"OOF MCC = {mcc}")

        self.logger.info("Final refit: finding best_iteration via 10% hold-out...")
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.10, random_state=42)
        tr_idx, val_idx = next(sss.split(X_arr, y_arr))
        probe = self._fit_model(
            self._build_model(),
            X_arr[tr_idx], y_arr[tr_idx],
            X_arr[val_idx], y_arr[val_idx],
        )
        best_iter = probe.best_iteration_
        self.logger.info(f"Early stopping found best_iteration = {best_iter}")

        final_params = {**PARAMS, 'n_estimators': best_iter}
        m = FLLGBMWrapper(**final_params, objective=FLLGBM(self.gamma, self.alpha))
        m.fit(X_arr, y_arr)
        self.logger.info("Final model trained on full dataset.")

        joblib.dump(m, ARTIFACTS_DIR / f"{name}.joblib")
        res = {
            'threshold': float(best_t),
            'oof_recall': float(r),
            'oof_precision': float(p),
            'oof_f2': float(best_f2),
            'oof_mcc': float(mcc),
            'mean_fold_recall': float(mean_recall),
            'gamma': float(self.gamma),
            'alpha': float(self.alpha),
            'best_iteration': int(best_iter),
            'n_trials': N_TRIALS,
            'train_attack_rate': float(attack_rate),
        }
        with open(ARTIFACTS_DIR / f"{name}_res.json", 'w') as f:
            json.dump(res, f, indent=2)
        self.logger.info(f"{name}: {res}")
        return res


if __name__ == '__main__':
    trainer = FLLGBMTrainer(artifacts_dir=ARTIFACTS_DIR, logger=logger)
    trainer.train()
