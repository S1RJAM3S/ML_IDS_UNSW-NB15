import warnings
from abc import ABC, abstractmethod
from pathlib import Path
import json
from sklearn.metrics import recall_score, precision_score, f1_score, fbeta_score, matthews_corrcoef, confusion_matrix
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from utils import make_logger, load_test
 
warnings.filterwarnings('ignore')

THRESHOLD_STEPS = 200
LABEL_RECALL_TARGET = 0.99
LABEL_PRECISION_TARGET = 0.80

class LabelEvaluator(ABC):
    def __init__(self, artifacts_dir: Path, report_dir: Path, logger, name: str):
        self.artifacts_dir = artifacts_dir
        self.report_dir = report_dir
        self.fig_dir = report_dir / 'figures'
        self.logger = logger
        self.name = name
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.fig_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def _load_model(self):
        pass

    @abstractmethod
    def _predict_proba(self, model, X: np.ndarray) -> ndarray:
        pass

    def _load_train_results(self) -> dict:
        path = self.artifacts_dir / f'{self.name}_res.json'
        assert path.exists(), f"Training results not found at {path}"
        with open(path) as f:
            return json.load(f)

    def _extra_report(self, model, X_arr, y_true, probs, report) -> dict:
        return {} # Override for extra report :>

    def _eval_at_threshold(self, y_true, probs, threshold, label) -> dict:
        preds = (probs >= threshold).astype(int)
        r = recall_score(y_true, preds, zero_division=0)
        p = precision_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)
        f2 = fbeta_score(y_true, preds, beta=2, average='binary', zero_division=0)
        mcc = matthews_corrcoef(y_true, preds)
        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
        fpr = fp / (fp + tn + 1e-9)
        fnr = fn / (fn + tp + 1e-9)

        self.logger.info(f"TP = {tp} - FP = {fp} - TN = {tn} - FN = {fn}")
        self.logger.info(f"Recall = {r} [Target >= {LABEL_RECALL_TARGET} {':>' if r >= LABEL_RECALL_TARGET else ':<'}]")
        self.logger.info(f"Precision = {p} [Target >= {LABEL_PRECISION_TARGET} {':>' if p >= LABEL_PRECISION_TARGET else ':<'}]")
        self.logger.info(f"MCC = {mcc}")
        self.logger.info(f"FPR = {fpr} ({fp} packets)")
        self.logger.info(f"FNR = {fnr} ({fp} packets)")
