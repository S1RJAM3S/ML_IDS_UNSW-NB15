import warnings
from abc import ABC, abstractmethod
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    recall_score, precision_score, f1_score, fbeta_score,
    matthews_corrcoef, confusion_matrix,
    average_precision_score, roc_auc_score,
    precision_recall_curve, classification_report, ConfusionMatrixDisplay,
)
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from utils import make_logger, load_test

warnings.filterwarnings('ignore')

THRESHOLD_STEPS = 200
LABEL_RECALL_TARGET = 0.98
LABEL_PRECISION_TARGET = 0.90


class LabelEvaluator(ABC):
    STAGE: str = 'label'

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
    def _predict_proba(self, model, X: np.ndarray) -> np.ndarray:
        pass

    def _load_train_results(self) -> dict:
        path = self.artifacts_dir / f'{self.name}_res.json'
        assert path.exists(), f"Training results not found at {path}"
        with open(path) as f:
            return json.load(f)

    def _extra_report(self, model, X_arr, y_true, probs, report) -> dict:
        return {}  # Override for extra diagnostics :>

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
        self.logger.info(f"[{label}] TP={tp} FP={fp} TN={tn} FN={fn}")
        self.logger.info(f"[{label}] Recall={r:.6f} [Target >= {LABEL_RECALL_TARGET} {':>' if r >= LABEL_RECALL_TARGET else ':<'}]")
        self.logger.info(f"[{label}] Precision={p:.6f} [Target >= {LABEL_PRECISION_TARGET} {':>' if p >= LABEL_PRECISION_TARGET else ':<'}]")
        self.logger.info(f"[{label}] MCC={mcc:.6f}")
        self.logger.info(f"[{label}] FPR={fpr:.6f} ({fp} packets)")
        self.logger.info(f"[{label}] FNR={fnr:.6f} ({fn} packets)")
        return {
            'threshold': float(threshold),
            'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
            'recall': float(r), 'precision': float(p),
            'f1': float(f1), 'f2': float(f2),
            'mcc': float(mcc),
            'fpr': float(fpr), 'fnr': float(fnr),
            'recall_target': bool(r >= LABEL_RECALL_TARGET),
            'precision_target': bool(p >= LABEL_PRECISION_TARGET),
        }

    def _prevalence_adjusted_threshold(self, base_t: float, train_rate: float,
                                        test_rate: float) -> float:
        eps = 1e-9
        lo_t = np.log(base_t / (1.0 - base_t + eps))
        lo_shift = (np.log(test_rate / (1.0 - test_rate + eps))
                    - np.log(train_rate / (1.0 - train_rate + eps)))
        adjusted_t = float(np.clip(1.0 / (1.0 + np.exp(-(lo_t - lo_shift))), 0.01, 0.99))
        self.logger.info(
            f"Prevalence adjustment (diagnostic): base_t={base_t:.4f} "
            f"train={train_rate:.4f} test={test_rate:.4f} → adjusted_t={adjusted_t:.4f}"
        )
        return adjusted_t

    def _eval_fn_by_category(self, y_true: np.ndarray, probs: np.ndarray,
                              y_cat, threshold: float) -> dict:
        if y_cat is None:
            return {}
        preds = (probs >= threshold).astype(int)
        y_cat_arr = np.array(y_cat)
        result = {}
        self.logger.info("FN breakdown by attack_cat:")
        attack_mask = y_true == 1
        attack_cats = y_cat_arr[attack_mask]
        attack_preds = preds[attack_mask]
        attack_probs = probs[attack_mask]
        for cat in sorted(set(attack_cats)):
            cat_mask = attack_cats == cat
            total = int(cat_mask.sum())
            fn = int((attack_preds[cat_mask] == 0).sum())
            fn_rate = fn / (total + 1e-9)
            median_score = float(np.median(attack_probs[cat_mask]))
            result[cat] = {'total': total, 'fn': fn, 'fn_rate': float(fn_rate), 'median_score': median_score}
            self.logger.info(f"{cat}: total={total}  FN={fn} FN_rate={fn_rate}  median_score={median_score}")
        cats = list(result.keys())
        fn_rates = [result[c]['fn_rate'] for c in cats]
        colours = ['#E24B4A' if r > 0.05 else '#1D9E75' for r in fn_rates]
        fig, ax = plt.subplots(figsize=(10, max(4, len(cats) * 0.5)))
        ax.barh(cats, fn_rates, color=colours)
        ax.axvline(0.05, color='gray', ls='--', lw=0.8, label='5% threshold')
        ax.set_xlabel('FN rate')
        ax.set_title(f'FN rate by attack category - {self.name}')
        ax.legend()
        ax.grid(axis='x', alpha=0.4)
        plt.tight_layout()
        plt.savefig(self.fig_dir / 'fn_by_category.png', dpi=100)
        plt.close()
        self.logger.info(f"Figure: {self.fig_dir / 'fn_by_category.png'}")
        return result

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
            rs.append(r); ps.append(p); f2s.append(f2); mccs.append(mcc)
            if r >= LABEL_RECALL_TARGET and p >= LABEL_PRECISION_TARGET and f2 > best_f2:
                best_f2 = f2
                best_f2_t = t
        if best_f2 < 0:
            self.logger.warning(f"Threshold sweep: no threshold met recall >= {LABEL_RECALL_TARGET} AND precision >= {LABEL_PRECISION_TARGET} simultaneously. :<")
        else:
            self.logger.info(f"Best F2 Threshold = {best_f2_t:.4f} -> F2 = {best_f2:.6f}")
        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        fig.suptitle(f"Threshold Sweep - {self.name}")
        colours = plt.colormaps['viridis'](np.linspace(0, 1, 4))
        for ax, vals, title, colour in zip(
            axes.flat, [rs, ps, f2s, mccs], ['Recall', 'Precision', 'F2', 'MCC'], colours
        ):
            ax.plot(thresholds, vals, color=colour)
            if best_f2 >= 0:
                ax.axvline(best_f2_t, color='r', ls='--', label=f'Best F2 t={best_f2_t:.3f}')
                ax.legend(fontsize=8)
            ax.set_title(title)
            ax.set_xlabel('Threshold')
            ax.grid(True)
        plt.tight_layout()
        plt.savefig(self.fig_dir / 'threshold_sweep.png', dpi=100)
        plt.close()
        self.logger.info(f"Figure: {self.fig_dir / 'threshold_sweep.png'}")
        return {'best_f2_threshold': float(best_f2_t), 'best_f2': float(best_f2)}

    def _eval_pr_curve(self, y_true, probs) -> dict:
        pr_auc = average_precision_score(y_true, probs)
        roc_auc = roc_auc_score(y_true, probs)
        ps, rs, _ = precision_recall_curve(y_true, probs)
        baseline = float(y_true.mean())
        self.logger.info(f"PR-AUC = {pr_auc:.6f}")
        self.logger.info(f"ROC-AUC = {roc_auc:.6f}")
        fig, ax = plt.subplots()
        ax.plot(rs, ps, color='b', label=f'{self.name} (PR-AUC={pr_auc:.4f})')
        ax.axhline(baseline, color='gray', label=f'Baseline={baseline:.3f}')
        ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
        ax.set_title(f"Precision-Recall Curve - {self.name}")
        ax.legend(); ax.grid(True)
        plt.tight_layout()
        plt.savefig(self.fig_dir / 'pr_curve.png', dpi=100)
        plt.close()
        self.logger.info(f"Figure: {self.fig_dir / 'pr_curve.png'}")
        return {'pr_auc': float(pr_auc), 'roc_auc': float(roc_auc), 'baseline': baseline}

    def _eval_score_dist(self, y_true, probs, threshold, adjusted_threshold=None) -> dict:
        p_normal = probs[y_true == 0]
        p_attack = probs[y_true == 1]
        self.logger.info(f"NORMAL: Mean={p_normal.mean():.4f} STD={p_normal.std():.4f} Median={np.median(p_normal):.4f}")
        self.logger.info(f"ATTACK: Mean={p_attack.mean():.4f} STD={p_attack.std():.4f} Median={np.median(p_attack):.4f}")
        fig, ax = plt.subplots()
        ax.hist(p_normal, bins=80, alpha=0.6, color='blue', label='Normal', density=True)
        ax.hist(p_attack, bins=80, alpha=0.6, color='red', label='Attack', density=True)
        ax.axvline(threshold, color='black', ls='--', label=f"OOF t={threshold:.4f} [primary]")
        if adjusted_threshold is not None:
            ax.axvline(adjusted_threshold, color='orange', ls=':', lw=1.5, label=f"Adjusted t={adjusted_threshold:.4f} [diagnostic]")
        ax.set_xlabel('Predicted Attack Probability'); ax.set_ylabel('Density')
        ax.set_title(f"Score Distribution - {self.name}")
        ax.legend(); ax.grid(True)
        plt.tight_layout()
        plt.savefig(self.fig_dir / 'score_dist.png', dpi=100)
        plt.close()
        self.logger.info(f"Figure: {self.fig_dir / 'score_dist.png'}")
        return {
            'normal_mean': float(p_normal.mean()), 'normal_median': float(np.median(p_normal)),
            'normal_std': float(p_normal.std()),
            'attack_mean': float(p_attack.mean()), 'attack_median': float(np.median(p_attack)),
            'attack_std': float(p_attack.std()),
        }

    def _eval_confusion(self, y_true, probs, threshold, label='OOF') -> dict:
        preds = (probs >= threshold).astype(int)
        cm = confusion_matrix(y_true, preds)
        self.logger.info(f"[{label}]\n{classification_report(y_true, preds, target_names=['Normal', 'Attack'], zero_division=0)}")
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay(cm, display_labels=['Normal', 'Attack']).plot(ax=ax, colorbar=True, cmap='Blues')
        ax.set_title(f"Confusion Matrix [{label}] - {self.name}")
        plt.tight_layout()
        fname = f'confusion_{label.lower().replace(" ", "_")}.png'
        plt.savefig(self.fig_dir / fname, dpi=100)
        plt.close()
        self.logger.info(f"Figure: {self.fig_dir / fname}")
        tn, fp, fn, tp = cm.ravel()
        return {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}

    def _eval_generalisation_gap(self, train_results, test_metrics):
        self.logger.info("Generalisation gaps (OOF threshold):")
        for key in ['recall', 'precision', 'f2', 'mcc']:
            oof = train_results.get(f'oof_{key}')
            test = test_metrics.get(key)
            if oof is not None and test is not None:
                gap = test - oof
                self.logger.info(f"{key}: OOF={oof:.4f} Test={test:.4f} - Gap={gap:+.4f} [{'OVERFIT' if gap < -0.03 else ':>'}]")

    def evaluate(self):
        self.logger.info(f"{self.name.upper()} - EVALUATION")
        model = self._load_model()
        train_results = self._load_train_results()
        X_test, y_label, y_cat = load_test(stage=self.STAGE)
        X_arr = X_test.values.astype(np.float64)
        y_true = y_label.values.astype(int)
        test_rate = float(y_true.mean())
        self.logger.info(f"Test shape: {X_test.shape}")
        self.logger.info(f"Attack rate: {test_rate:.4f}")

        train_rate = train_results.get('train_attack_rate')
        adjusted_threshold = None
        if train_rate:
            shift = abs(test_rate - train_rate)
            if shift > 0.05:
                self.logger.warning(
                    f"Distribution shift: Train={train_rate:.4f} Test={test_rate:.4f} Diff={shift:.4f}"
                )
                adjusted_threshold = self._prevalence_adjusted_threshold(
                    float(train_results['threshold']), train_rate, test_rate
                )

        probs = self._predict_proba(model, X_arr)
        oof_threshold = float(train_results['threshold'])

        report = {
            'model': self.name,
            'training_oof': {
                k: train_results[k]
                for k in ['oof_recall', 'oof_precision', 'oof_f2', 'oof_mcc', 'threshold']
                if k in train_results
            },
        }

        report['test_at_threshold'] = self._eval_at_threshold(
            y_true, probs, oof_threshold, label='OOF threshold'
        )

        if adjusted_threshold is not None:
            report['test_at_adjusted_threshold'] = self._eval_at_threshold(
                y_true, probs, adjusted_threshold, label='Adjusted threshold'
            )
            report['prevalence_adjustment'] = {
                'train_rate': float(train_rate), 'test_rate': test_rate,
                'oof_threshold': oof_threshold, 'adjusted_threshold': adjusted_threshold,
            }

        report['threshold_sweep'] = self._eval_threshold_sweep(y_true, probs)
        report['pr_roc_auc'] = self._eval_pr_curve(y_true, probs)
        report['score_distribution'] = self._eval_score_dist(
            y_true, probs, oof_threshold, adjusted_threshold=adjusted_threshold
        )
        report['confusion'] = self._eval_confusion(y_true, probs, oof_threshold, label='OOF threshold')
        self._eval_generalisation_gap(train_results, report['test_at_threshold'])
        report['fn_by_category'] = self._eval_fn_by_category(
            y_true, probs,
            y_cat.values if y_cat is not None else None,
            oof_threshold,
        )
        extra = self._extra_report(model, X_arr, y_true, probs, report)
        report.update(extra)

        out = self.report_dir / 'report.json'
        with open(out, 'w') as f:
            json.dump(report, f, indent=2)
        self.logger.info(f"{self.name}: saved report to {out}")
