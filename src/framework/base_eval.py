import warnings
from abc import ABC, abstractmethod
from pathlib import Path
import json
import matplotlib
import numpy as np
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score, precision_score, f1_score, fbeta_score, matthews_corrcoef, confusion_matrix, average_precision_score, roc_auc_score, precision_recall_curve, classification_report, ConfusionMatrixDisplay
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from utils import make_logger, load_test
 
warnings.filterwarnings('ignore')

THRESHOLD_STEPS = 200
LABEL_RECALL_TARGET = 0.98
LABEL_PRECISION_TARGET = 0.90

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

        return {
            'threshold': float(threshold),
            'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
            'recall': float(r), 'precision': float(p),
            'f1': float(f1), 'f2': float(f2),
            'mcc': float(mcc),
            'fpr': float(fpr), 'fnr': float(fnr),
            'recall_target': bool(r >= LABEL_RECALL_TARGET)
        }

    def _eval_threshold_sweep(self, y_true, probs) -> dict:
        thresholds = np.linspace(0.01, 0.99, THRESHOLD_STEPS)
        rs, ps, f2s, mccs = [], [], [], []
        best_f2_t, best_f2 = 0.5, -1.0
        
        for t in thresholds:
            preds = (probs >= t).astype(int)
            r = recall_score(y_true, preds, zero_division=0)
            p = precision_score(y_true, preds, zero_division=0)
            f2 = fbeta_score(y_true, preds, beta=2, average='binary', zero_division=0)
            mcc = matthews_corrcoef(y_true, preds)
            rs.append(r)
            ps.append(p)
            f2s.append(f2)
            mccs.append(mcc)
            if r >= LABEL_RECALL_TARGET and p >= LABEL_PRECISION_TARGET and f2 > best_f2:
                best_f2 = f2
                best_f2_t = t

        self.logger.info(f"Best F2 Threshold = {best_f2_t} -> F2 = {best_f2}")
        fig, axes = plt.subplots(2, 2)
        fig.suptitle(f"Threshold Sweep - {self.name}")
        colours = plt.colormaps['viridis'](np.linspace(0, 1, 4))
        for ax, vals, title, colour in zip(axes.flat, [rs, ps, f2s, mccs], ['Recall', 'Precision', 'F2', 'MCC'], colours):
            ax.plot(thresholds, vals, color=colour)
            ax.axvline(best_f2_t, color='r', ls='--', label=f'Best F2 t={best_f2_t}')
            ax.set_title(title)
            ax.set_xlabel('Threshold')
            ax.grid(True)
        plt.tight_layout()
        plt.savefig(self.fig_dir / 'threshold_sweep.png')
        plt.close()
        self.logger.info(f"Figure: {self.fig_dir / 'threshold_sweep.png'}")
        
        return {
            'best_f2_threshold': float(best_f2_t),
            'best_f2': float(best_f2)
        }

    def _eval_pr_curve(self, y_true, probs) -> dict:
        pr_auc = average_precision_score(y_true, probs)
        roc_auc = roc_auc_score(y_true, probs)
        ps, rs, _ = precision_recall_curve(y_true, probs)
        baseline = float(y_true.mean())
        self.logger.info(f"PR-AUC = {pr_auc}")
        self.logger.info(f"ROC-AUC = {roc_auc}") # Less reliable under imbalanced :<

        fig, ax = plt.subplots()
        ax.plot(rs, ps, color='b', label=f'{self.name} (PR-AUC={pr_auc})')
        ax.axhline(baseline, color='gray', label=f'Baseline={baseline}')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(f"Precision-Recall Curve - {self.name}")
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(self.fig_dir / 'pr_curve.png')
        plt.close()
        self.logger.info(f"Figure: {self.fig_dir / 'pr_curve.png'}")

        return {
            'pr_auc': float(pr_auc),
            'roc_auc': float(roc_auc),
            'baseline': float(baseline)
        }

    def _eval_score_dist(self, y_true, probs, threshold) -> dict:
        p_normal = probs[y_true == 0]
        p_attack = probs[y_true == 1]
        self.logger.info(f"NORMAL: Mean = {p_normal.mean()} - STD = {p_normal.std()} - Median = {np.median(p_normal)}")
        self.logger.info(f"ATTACK: Mean = {p_attack.mean()} - STD = {p_attack.std()} - Median = {np.median(p_attack)}")

        fig, ax = plt.subplots()
        ax.hist(p_normal, bins=80, alpha=0.6, color='blue', label='Normal', density=True)
        ax.hist(p_attack, bins=80, alpha=0.6, color='red', label='Attack', density=True)
        ax.axvline(threshold, color='black', ls='--', label=f"Threshold={threshold}")
        ax.set_xlabel('Predicted Attack Probability')
        ax.set_ylabel('Density')
        ax.set_title(f"Score Distribution - {self.name}")
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(self.fig_dir / 'score_dist.png')
        plt.close()
        self.logger.info(f"Figure: {self.fig_dir / 'score_dist.png'}")

        return {
            'normal_mean': float(p_normal.mean()),
            'normal_median': float(np.median(p_normal)),
            'attack_mean': float(p_attack.mean()),
            'attack_median': float(np.median(p_attack))
        }

    def _eval_confusion(self, y_true, probs, threshold) -> dict:
        preds = (probs >= threshold).astype(int)
        cm = confusion_matrix(y_true, preds)
        self.logger.info(f"\n{classification_report(y_true, preds, target_names=['Normal', 'Attack'], zero_division=0)}")

        fig, ax = plt.subplots()
        ConfusionMatrixDisplay(cm, display_labels=['Normal', 'Attack']).plot(ax=ax, colorbar=True, cmap='Blues')
        ax.set_title(f"Confusion Matrix - {self.name}")
        plt.tight_layout()
        plt.savefig(self.fig_dir / 'confusion.png')
        plt.close()
        self.logger.info(f"Figure: {self.fig_dir / 'confusion.png'}")
        
        tn, fp, fn, tp = cm.ravel()
        return { 'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp) }

    def _eval_generalisation_gap(self, train_results, test_metrics):
        for key in ['recall', 'precision', 'f2', 'mcc']:
            oof = train_results.get(f'oof_{key}')
            test = test_metrics.get(key)
            if oof is not None and test is not None:
                gap = test - oof
                self.logger.info(f"{key}: OOF = {oof} - Test = {test} -> Gap = {gap} [{'OVERFIT' if gap < -0.03 else ':>'}]")

    def evaluate(self):
        self.logger.info(f"{self.name.upper()} - EVALUATION")
        model = self._load_model()
        train_results = self._load_train_results()
        X_test, y_label, _ = load_test()
        X_arr = X_test.values.astype(np.float64)
        y_true = y_label.values.astype(int)
        self.logger.info(f"Test shape: {X_test.shape}")
        self.logger.info(f"Attack rate: {y_true.mean()}")
        train_rate = train_results.get('train_attack_rate')
        if train_rate:
            shift = abs(y_true.mean() - train_rate)
            if shift > 0.05:
                self.logger.warning(f"Distribution shift: Train = {train_rate} - Test = {y_true.mean()} -> Diff = {shift}")

        probs = self._predict_proba(model, X_arr)
        threshold = float(train_results['threshold'])
        report = {
            'model': self.name,
            'training_oof': {
                k: train_results[k] for k in ['oof_recall', 'oof_precision', 'oof_f2', 'oof_mcc', 'threshold'] if k in train_results
            }
        }
        report['test_at_threshold'] = self._eval_at_threshold(y_true, probs, threshold, self.name)
        report['threshold_sweep'] = self._eval_threshold_sweep(y_true, probs)
        report['pr_roc_auc'] = self._eval_pr_curve(y_true, probs)
        report['score_distribution'] = self._eval_score_dist(y_true, probs, threshold)
        report['confusion'] = self._eval_confusion(y_true, probs, threshold)
        self._eval_generalisation_gap(train_results, report['test_at_threshold'])
        extra = self._extra_report(model, X_arr, y_true, probs, report)
        report.update(extra)

        out = self.report_dir / 'report.json'
        with open(out, 'w') as f:
            json.dump(report, f)
        self.logger.info(f"{self.name}: {report}")
