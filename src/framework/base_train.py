import warnings
from abc import ABC, abstractmethod
from pathlib import Path
import joblib
import json
import numpy as np
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from imblearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
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
    STAGE: str = 'label'

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

    def _fit_model(self, model, X_tr, y_tr, X_val, y_val, sample_weight=None):
        model.fit(X_tr, y_tr)
        return model

    def _cv_loop(self, X_arr: np.ndarray, y_arr: np.ndarray, sample_weights: np.ndarray = None) -> tuple:
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
        oof_probs = np.zeros(len(y_arr), dtype=np.float64)
        fold_recalls, fold_precisions = [], []
        for fold, (tr_idx, val_idx) in enumerate(skf.split(X_arr, y_arr)):
            X_tr, X_val = X_arr[tr_idx], X_arr[val_idx]
            y_tr, y_val = y_arr[tr_idx], y_arr[val_idx]
            fold_sw = sample_weights[tr_idx] if sample_weights is not None else None
            model = self._build_model()
            model = self._fit_model(model, X_tr, y_tr, X_val, y_val, sample_weight=fold_sw)
            probs = self._predict_proba(model, X_val)
            oof_probs[val_idx] = probs
            preds = (probs >= 0.5).astype(int)
            fold_recalls.append(recall_score(y_val, preds, zero_division=0))
            fold_precisions.append(precision_score(y_val, preds, zero_division=0))
            self.logger.info(f" Fold {fold+1}/{N_SPLITS} - Recall@0.5 = {fold_recalls[-1]} - Precision@0.5 = {fold_precisions[-1]}")
        mean_recall = float(np.mean(fold_recalls))
        mean_precision = float(np.mean(fold_precisions))
        self.logger.info(f"Mean OOF Recall@0.5 = {mean_recall} - Mean OOF Precision@0.5 = {mean_precision}")
        return oof_probs, mean_recall

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
            self.logger.warning("No threshold met both targets; fallback to max-recall threshold :<")
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

    def train(self, name: str = 'model'):
        X, y_label, _ = load_train(stage=self.STAGE)
        X_arr = X.values.astype(np.float64)
        y_arr = y_label.values.astype(int)
        self.logger.info(f"Train shape: {X.shape}")
        self.logger.info(f"Attack rate: {y_arr.mean()}")

        oof_probs, mean_recall = self._cv_loop(X_arr, y_arr)
        self.logger.info(f"Mean OOF Recall@0.5: {mean_recall}")

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
            'precision_target': LABEL_PRECISION_TARGET,
        }
        self._save(model, res, name)
        self.logger.info(f"{name}: {res}")
        return res

class CatTrainer(ABC):
    STAGE: str = 'cat'

    def __init__(self, artifacts_dir: Path, logger, smote_target_ratio: float = 0.3, smote_max_multiplier: int = 8): # DEFAULT
        self.artifacts_dir = artifacts_dir
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger
        self.smote_target_ratio = smote_target_ratio
        self.smote_max_multiplier = smote_max_multiplier

    @abstractmethod
    def _build_model(self):
        pass

    @abstractmethod
    def _predict_proba(self, model, X: np.ndarray) -> np.ndarray:
        pass

    def _fit_model(self, model, X_tr, y_tr, X_val, y_val, sample_weight=None):
        model.fit(X_tr, y_tr)
        return model

    def _get_smote(self, y_train: np.ndarray):
        cnts = Counter(y_train)
        majority = max(cnts.values())
        target = int(majority * self.smote_target_ratio)
        strat = {}
        for cls, cnt in cnts.items():
            safe = min(target, cnt * self.smote_max_multiplier)
            if safe > cnt:
                strat[cls] = safe
        self.logger.info(f"SMOTE targets (ratio = {self.smote_target_ratio} - Max multiplier = {self.smote_max_multiplier} - Majority = {majority} - Target = {target}: " + ', '.join(f"{c}:{n}" for c, n in sorted(strat.items()) if n > cnts[c]))
        return Pipeline([
            ('smote', SMOTE(sampling_strategy=strat, k_neighbors=3, random_state=42)),
            ('tomek', TomekLinks(sampling_strategy='majority'))
        ])

    def _cv_loop(self, X_arr: np.ndarray, y_arr: np.ndarray) -> tuple:
        le = LabelEncoder()
        y_enc = le.fit_transform(y_arr)
        classes = list(le.classes_)
        n_cls = len(classes)
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
        oof_probs = np.zeros((len(y_arr), n_cls), dtype=np.float64)
        fold_recalls = []
        for fold, (tr_idx, val_idx) in enumerate(skf.split(X_arr, y_enc)):
            X_tr, X_val = X_arr[tr_idx], X_arr[val_idx]
            y_tr, y_val = y_arr[tr_idx], y_arr[val_idx]
            smote = self._get_smote(y_tr)
            if smote is not None:
                X_tr_res, y_tr_res = smote.fit_resample(X_tr, y_tr)
            else:
                X_tr_res, y_tr_res = X_tr, y_tr
                self.logger.info(f"Skipped SMOTE")
            model = self._fit_model(self._build_model(), X_tr_res, y_tr_res, X_val, y_val)
            probs = self._predict_proba(model, X_val)
            oof_probs[val_idx] = probs
            preds = np.array(classes)[probs.argmax(axis=1)]
            r = recall_score(y_val, preds, average='macro', zero_division=0) # Macro Recall!
            fold_recalls.append(r)
            self.logger.info(f"Fold {fold+1}/{N_SPLITS} - Macro Recall = {r}")
        mean_recall = float(np.mean(fold_recalls))
        self.logger.info(f"Mean OOF Macro Recall = {mean_recall}")
        return oof_probs, mean_recall, classes

    def _save(self, model, oof_probs: np.ndarray, classes: list, res: dict, name: str):
        joblib.dump(model, self.artifacts_dir / f'{name}.joblib')
        np.save(self.artifacts_dir / f'{name}_oof.npy', oof_probs)
        with open(self.artifacts_dir / f'{name}_res.json', 'w') as f:
            json.dump(res, f)
        self.logger.info(f"RESULT ({name}): {res}")

    def train(self, name: str, label_oof_path: Path = None, label_threshold: float = None):
        X, _, y_cat = load_train(stage=self.STAGE)
        X_arr = X.values.astype(np.float64)
        y_arr = y_cat.values
        if label_oof_path is not None and label_threshold is not None:
            label_oof = np.load(label_oof_path)
            mask = label_oof >= label_threshold
            X_arr = X_arr[mask]
            y_arr = y_arr[mask]
            self.logger.info(f"Applied Label filter :>")
        self.logger.info(f"Train shape: {X.shape}")
        self.logger.info(f"Class distribution:")
        for cls, cnt in sorted(Counter(y_arr).items()):
            self.logger.info(f"{cls}: {cnt}")

        oof_probs, mean_recall, classes = self._cv_loop(X_arr, y_arr)
        self.logger.info(f"Mean OOF Macro Recall: {mean_recall}")
        preds = np.array(classes)[oof_probs.argmax(axis=1)]
        per_cls = {}
        for cls in classes:
            mask = y_arr == cls
            if mask.sum() == 0:
                continue
            r = recall_score(y_arr[mask], preds[mask], average='micro', zero_division=0)
            self.logger.info(f"{cls}: {r}")
            per_cls[cls] = float(r)
        model = self._fit_model(self._build_model(), X_arr, y_arr, None, None)
        res = {
            'mean_oof_macro_recall': float(mean_recall),
            'per_class_oof_recall': per_cls,
            'classes': classes,
            'n_splits': N_SPLITS,
            'smote_target_ratio': self.smote_target_ratio,
            'smote_max_multiplier': self.smote_max_multiplier
        }
        self._save(model, oof_probs, classes, res, name)
        self.logger.info(f"RESULT ({name}): {res}")
        return res
