import warnings
from abc import ABC, abstractmethod
from pathlib import Path
import joblib
import json
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import recall_score, precision_score, matthews_corrcoef, fbeta_score
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from utils import make_logger, load_train

warnings.filterwarnings('ignore')

N_SPLITS = 10
THRESHOLD_STEPS = 200
LABEL_RECALL_TARGET = 0.98
LABEL_PRECISION_TARGET = 0.90

class LabelTrainer(ABC):
    def __init__(self, artifacts_dir: Path, logger):
        self.artifacts_dir = artifacts_dir
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger

    @abstractmethod
    def _build_model(self):
        pass

    @abstractmethod
    def _predict_proba(self, model, X: np.ndarray) -> np.ndarray:
        pass

    def _fit_model(self, model, X_tr, y_tr, X_val, y_val):
        model.fit(X_tr, y_tr)
        return model

    def _cv_loop(self, X_arr: np.ndarray, y_arr: np.ndarray) -> tuple:
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
        oof_probs = np.zeros(len(y_arr), dtype=np.float64)
        fold_recalls = []
        for fold, (tr_idx, val_idx) in enumerate(skf.split(X_arr, y_arr)):
            X_tr, X_val = X_arr[tr_idx], X_arr[val_idx]
            y_tr, y_val = y_arr[tr_idx], y_arr[val_idx]
            # NOTE: No more SMOTE here, too much Attack already :<
            model = self._build_model()
            model = self._fit_model(model, X_tr, y_tr, X_val, y_val)
            probs = self._predict_proba(model, X_val)
            oof_probs[val_idx] = probs
            fold_recalls.append(recall_score(y_val, (probs >= 0.5).astype(int), zero_division=0))
            self.logger.info(f" Fold {fold+1}/{N_SPLITS} - Recall = {fold_recalls[-1]}")
        return oof_probs, float(np.mean(fold_recalls))

    def _find_threshold(self, y_arr: np.ndarray, oof_probs: np.ndarray) -> tuple:
        thresholds = np.linspace(0.01, 0.99, THRESHOLD_STEPS)
        best_t = 0.5
        best_f2 = -1.0
        for t in thresholds:
            preds = (oof_probs >= t).astype(int)
            r = recall_score(y_arr, preds, zero_division=0)
            p = precision_score(y_arr, preds, zero_division=0)
            if r >= LABEL_RECALL_TARGET and p >= LABEL_PRECISION_TARGET:
                f2 = fbeta_score(y_arr, preds, beta=2, average='binary', zero_division=0)
                if f2 > best_f2:
                    best_f2 = f2
                    best_t = t

        if best_f2 < 0:
            self.logger.warning(f"No threshold met targets, fallback to  max recall threshold :<")
            best_t = float(thresholds[np.argmax([recall_score(y_arr, (oof_probs >= t).astype(int), zero_division=0) for t in thresholds])])
            best_pred = (oof_probs >= best_t).astype(int)
            best_f2 = fbeta_score(y_arr, best_pred, beta=2, average='binary', zero_division=0)

        return float(best_t), float(best_f2)

    def _save(self, model, res: dict, name: str):
        model_path = self.artifacts_dir / f'{name}.joblib'
        res_path = self.artifacts_dir / f'{name}_res.json'
        joblib.dump(model, model_path)
        with open(res_path, 'w') as f:
            json.dump(res, f)
        self.logger.info(f"Saved model ({model_path}) and result ({res_path})")

    def train(self, name: str = 'model'): # Default train, maybe override?
        X, y_label, _ = load_train()
        X_arr = X.values.astype(np.float64)
        y_arr = y_label.values.astype(int)
        self.logger.info(f"Train shape: {X.shape}")
        self.logger.info(f"Attack rate: {y_arr.mean()}")
        
        oof_probs, mean_recall = self._cv_loop(X_arr, y_arr)
        self.logger.info(f"Mean OOF Recall: {mean_recall}")

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

        model = self._fit_model(self._build_model(), X_arr, y_arr, None, None)
        res = {
            'threshold': float(best_t),
            'oof_recall': float(r),
            'oof_precision': float(p),
            'oof_f2': float(best_f2),
            'oof_mcc': float(mcc),
            'mean_fold_recall': float(mean_recall),
            'train_attack_rate': float(y_arr.mean()),
            'n_splits': N_SPLITS,
            'recall_target': LABEL_RECALL_TARGET,
            'precision_target': LABEL_PRECISION_TARGET
        }
        self._save(model, res, name)
        self.logger.info(f"{name}: {res}")
        return res
