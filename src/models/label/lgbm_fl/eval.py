import warnings
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.calibration import calibration_curve
import sys
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from framework.base_eval import LabelEvaluator
from utils import make_logger, load_test

from train import FLLGBM, FLLGBMEvalError, FLLGBMWrapper

warnings.filterwarnings('ignore')
 
PATH = Path(__file__).parent
ARTIFACTS_DIR = PATH / 'artifacts'
REPORT_DIR = PATH / 'report'
REPORT_DIR.mkdir(parents=True, exist_ok=True)

class FLLGBMEvaluator(LabelEvaluator):
    def _load_model(self) -> tuple:
        models = []
        for p, name in [(ARTIFACTS_DIR / 'lgbm_fl_c.joblib', 'lgbm_fl_c.joblib'), (ARTIFACTS_DIR / 'lgbm_fl.joblib', 'lgbm_fl.joblib')]:
            assert p.exists(), f"{name} not found :<"
            models.append(joblib.load(p))
        c, m = models
        return c, m

    def _predict_proba(self, model, X: np.ndarray) -> np.ndarray:
        c, _ = model
        return c.predict_proba(X)[:, 1]

    def _extra_report(self, model, X_arr, y_true, probs, report) -> dict:
        c, m = model
        probs_raw = m.predict_proba(X_arr)[:, 1]
        probs_cal = probs

        frac_raw, mean_raw = calibration_curve(y_true, probs_raw, n_bins=15, strategy='uniform')
        frac_cal, mean_cal = calibration_curve(y_true, probs_cal, n_bins=15, strategy='uniform')

        def ece(probs_in, frac, mean):
            cnts = np.histogram(probs_in, bins=15, range=(0, 1))[0]
            ws = cnts / (cnts.sum() + 1e-9)
            return float(np.sum(ws[:len(frac)] * np.abs(frac - mean)))

        ece_raw = ece(probs_raw, frac_raw, mean_raw)
        ece_cal = ece(probs_cal, frac_cal, mean_cal)
        self.logger.info(f"ECE Uncalibrated = {ece_raw}")
        self.logger.info(f"ECE Calibrated = {ece_cal}")
        self.logger.info(f"Improve? {ece_raw - ece_cal}")

        fix, ax = plt.subplots()
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        ax.plot(mean_raw, frac_raw, 's-', color='blue', label=f'Uncalibrated')
        ax.plot(mean_cal, frac_cal, 'o-', color='red', label=f'Calibrated')
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("True Attack Rate")
        ax.set_title("Calibration Curve - LightGBM Focal Loss")
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(self.fig_dir / 'calibration.png')
        plt.close()
        self.logger.info(f"Figure: {self.fig_dir / 'calibration.png'}")

        return {
            'calibration': {
                'ece_uncalibrated': ece_raw,
                'ece_calibrated': ece_cal,
                'ece_improvement': ece_raw - ece_cal
            }
        }

if __name__ == '__main__':
    logger = make_logger(__name__, str(REPORT_DIR / 'eval.log'))
    e = FLLGBMEvaluator(artifacts_dir=ARTIFACTS_DIR, report_dir=REPORT_DIR, logger=logger, name='lgbm_fl')
    e.evaluate()
