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

from train import FLLGBMWrapper, FLLGBM, FLLGBMEvalError

warnings.filterwarnings('ignore')

PATH = Path(__file__).parent
ARTIFACTS_DIR = PATH / 'artifacts'
REPORT_DIR = PATH / 'report'
REPORT_DIR.mkdir(parents=True, exist_ok=True)


class FLLGBMEvaluator(LabelEvaluator):
    STAGE = 'label'  # loads selected_label_cols from artifacts

    def _load_model(self) -> FLLGBMWrapper:
        p = ARTIFACTS_DIR / 'lgbm_fl.joblib'
        assert p.exists(), "lgbm_fl.joblib not found :<"
        return joblib.load(p)

    def _predict_proba(self, model, X: np.ndarray) -> np.ndarray:
        return model.predict_proba(X)[:, 1]

    def _extra_report(self, model, X_arr, y_true, probs, report) -> dict:
        frac, mean = calibration_curve(y_true, probs, n_bins=15, strategy='uniform')

        def ece(probs_in, frac_in, mean_in):
            cnts = np.histogram(probs_in, bins=15, range=(0, 1))[0]
            ws = cnts / (cnts.sum() + 1e-9)
            return float(np.sum(ws[:len(frac_in)] * np.abs(frac_in - mean_in)))

        ece_val = ece(probs, frac, mean)
        self.logger.info(f"ECE (raw model) = {ece_val:.6f}")

        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        ax.plot(mean, frac, 'o-', color='blue', label=f'Model (ECE={ece_val:.4f})')
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("True attack rate")
        ax.set_title("Calibration curve - LightGBM Focal Loss")
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(self.fig_dir / 'calibration.png', dpi=100)
        plt.close()
        self.logger.info(f"Figure: {self.fig_dir / 'calibration.png'}")

        return {'calibration': {'ece': ece_val}}


if __name__ == '__main__':
    logger = make_logger(__name__, str(REPORT_DIR / 'eval.log'))
    e = FLLGBMEvaluator(
        artifacts_dir=ARTIFACTS_DIR,
        report_dir=REPORT_DIR,
        logger=logger,
        name='lgbm_fl',
    )
    e.evaluate()
