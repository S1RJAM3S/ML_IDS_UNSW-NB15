import warnings
import json
import numpy as np
import joblib
import lightgbm as lgb
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import recall_score, precision_score, matthews_corrcoef
from sklearn.calibration import CalibratedClassifierCV
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from utils import make_logger, load_train, get_smote_label

warnings.filterwarnings('ignore')

PATH = Path(__file__).parent
ARTIFACTS_DIR = PATH / 'artifacts'
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
M_PATH = ARTIFACTS_DIR / 'lgbm_fl.joblib'
C_PATH = ARTIFACTS_DIR / 'lgbm_fl_c.joblib'
R_PATH = ARTIFACTS_DIR / 'lgbm_fl_r.json'

logger = make_logger(__name__, str(PATH / 'lgbm_fl.log'))

N_TRIALS = 30

# NOTE: Fixed PicklingError :<
class FLLGB:
    def __init__(self, gamma: float, alpha: float):
        self.gamma = gamma
        self.alpha = alpha
 
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray):
        y = y_true.astype(int)
        p = 1.0 / (1.0 + np.exp(-y_pred))
        p_t = np.where(y==1, p, 1.0 - p)
        alpha_t = np.where(y==1, self.alpha, 1.0 - self.alpha)
        grad = alpha_t * ((1 - p_t) ** self.gamma) * (self.gamma * p_t * np.log(p_t + 1e-9) + p_t - 1)
        grad = np.where(y==1, grad, -grad)
        hess = alpha_t * (1.0 - p_t) ** self.gamma * p * (1.0 - p)
        return grad, hess

class FLLGBEvalError:
    def __init__(self, gamma: float):
        self.gamma = gamma
 
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray):
        y = y_true.astype(int)
        p = 1.0 / (1.0 + np.exp(-y_pred))
        p_t = np.where(y==1, p, 1.0 - p)
        loss = -((1 - p_t) ** self.gamma) * np.log(p_t + 1e-9)
        return 'focal_loss', float(loss.mean()), False  # False = lower is better :>

class FLLGBMWrapper(lgb.LGBMClassifier): # CalibratedClassifierCV called predict_proba() -> 2D array!
    def predict_proba(self, X, **kwargs):
        raw = super().predict_proba(X, **kwargs)
        p = 1.0 / (1.0 + np.exp(-raw))
        return np.column_stack([1.0 - p, p])

PARAMS = {
    'n_estimators': 1000,
    'learning_rate': 0.05,
    'num_leaves': 64, # Dataset big!!!
    'max_depth': -1,
    'min_data_in_leaf': 20,
    'bagging_fraction': 0.8, # Imbalanced binary classification :>
    'feature_fraction': 0.8,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}
N_SPLITS = 10
RECALL = 0.99
PRECISION = 0.50 # Min precision -> attack_cat != FP
THRESHOLD_STEPS = 200

def _cv_loop(X_arr, y_arr, gamma: float, alpha: float) -> tuple:
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    p_oof = np.zeros(len(y_arr), dtype=np.float64)
    fold_recalls = []
    fl_lgb = FLLGB(gamma, alpha)
    fl_lgb_eval_error = FLLGBEvalError(gamma)
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_arr, y_arr)):
        X_tr, X_val = X_arr[tr_idx], X_arr[val_idx]
        y_tr, y_val = y_arr[tr_idx], y_arr[val_idx]
        smote = get_smote_label()
        X_tr_res, y_tr_res = smote.fit_resample(X_tr, y_tr)
        model = FLLGBMWrapper(**PARAMS, objective=fl_lgb)
        model.fit(X_tr_res, y_tr_res, eval_set=[(X_val, y_val)], eval_metric=fl_lgb_eval_error) 
        raw = model.predict_proba(X_val)[:, 1]
        p_oof[val_idx] = raw
        fold_recalls.append(recall_score(y_val, (raw >= 0.5).astype(int), zero_division=0))
    return p_oof, float(np.mean(fold_recalls))

def train():
    X, y_label, _ = load_train()
    X_arr = X.values.astype(np.float64)
    y_arr = y_label.values.astype(int)
    logger.info(f"Train shape: {X.shape} - Attack rate: {y_arr.mean():.6f}")
    # NOTE: attack > normal (label ONLY!!!)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def obj(trial):
        # Common (Google)
        gamma = trial.suggest_float('gamma', 1.0, 4.0, step=0.25)
        alpha = trial.suggest_float('alpha', 0.25, 0.75, step=0.05) # rate = ~0.6
        _, mean_recall = _cv_loop(X_arr, y_arr, gamma, alpha)
        logger.info(f"Trial {trial.number} - gamma = {gamma:.2f} - alpha = {alpha:.2f} - OOF Recall = {mean_recall:.6f}")
        return mean_recall

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(obj, n_trials=N_TRIALS)
    best_gamma = study.best_params['gamma']
    best_alpha = study.best_params['alpha']
    logger.info(f"Best params: gamma = {best_gamma:.2f} - alpha = {best_alpha:.2f} - OOF Recall = {study.best_value:.6f}")
    
    p_oof, mean_recall = _cv_loop(X_arr, y_arr, best_gamma, best_alpha)
    logger.info(f"Mean OOF Recall: {mean_recall:.6f}")
    thresholds = np.linspace(0.01, 0.99, THRESHOLD_STEPS)
    best_thresh = 0.5
    best_f2 = -1.0
    for t in thresholds:
        pred = (p_oof >= t).astype(int)
        rs = recall_score(y_arr, pred, zero_division=0)
        ps = precision_score(y_arr, pred, zero_division=0)
        if rs >= RECALL and ps >= PRECISION:
            f2 = (5 * ps * rs) / (4 * ps + rs + 1e-9)
            if f2 > best_f2:
                best_f2 = f2
                best_thresh = t
    best_pred = (p_oof >= best_thresh).astype(int)
    logger.info(f"Best threshold: {best_thresh:.4f}")
    logger.info(f"OOF Recall: {recall_score(y_arr, best_pred, zero_division=0):.4f}")
    logger.info(f"OOF Precision: {precision_score(y_arr, best_pred, zero_division=0):.4f}")
    logger.info(f"OOF F2: {best_f2:.4f}")
    logger.info(f"OOF MCC: {matthews_corrcoef(y_arr, best_pred):.4f}")
    if best_f2 < 0:
        logger.warning(":<")
        best_thresh = float(thresholds[np.argmax([recall_score(y_arr, (p_oof >= t).astype(int), zero_division=0) for t in thresholds])])
        best_pred = (p_oof >= best_thresh).astype(int)
    
    best_fl_lgb = FLLGB(best_gamma, best_alpha)
    m = FLLGBMWrapper(**PARAMS, objective=best_fl_lgb)
    m.fit(X_arr, y_arr)
    c = CalibratedClassifierCV(FLLGBMWrapper(**PARAMS, objective=best_fl_lgb), method='isotonic', cv=5)
    c.fit(X_arr, y_arr)

    joblib.dump(m, M_PATH)
    joblib.dump(c, C_PATH)
    res = {
        'threshold': round(float(best_thresh), 6),
        'oof_recall': round(float(recall_score(y_arr, best_pred, zero_division=0)), 6),
        'oof_precision': round(float(precision_score(y_arr, best_pred, zero_division=0)), 6),
        'oof_f2': round(float(best_f2), 6),
        'oof_mcc': round(float(matthews_corrcoef(y_arr, best_pred)), 6),
        'mean_fold_recall': round(float(mean_recall), 6),
        'gamma': best_gamma,
        'alpha': best_alpha,
        'n_trials': N_TRIALS,
        'recall': RECALL,
        'precision': PRECISION,
        'n_splits': N_SPLITS
    }
    with open(R_PATH, 'w') as f:
        json.dump(res, f, indent=2)
    logger.info(f"RESULTS: {res}")

if __name__ == '__main__':
    train()
