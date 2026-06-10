import warnings
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils import make_logger, load_test
sys.path.append(str(Path(__file__).parent.parent / 'src' / 'models' / 'label' / 'lgbm_fl'))
from train import FLLGBM, FLLGBMEvalError, FLLGBMWrapper  # noqa: F401

warnings.filterwarnings('ignore')

PATH = Path(__file__).resolve().parent
ROOT = PATH.parent
MODELS = ROOT / 'src' / 'models'
LABEL_DIR = MODELS / 'label'
CAT_DIR = MODELS / 'cat'
REPORT_DIR = PATH / 'report'
REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = REPORT_DIR / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

with open(LABEL_DIR / 'lgbm_fl' / 'artifacts' / 'lgbm_fl_res.json', 'r') as f:
    LABEL_THRESHOLD = float(json.load(f)['threshold'])

BASE_LEARNERS = [
    ('lgbm_gbdt', CAT_DIR / 'lgbm_gbdt' / 'artifacts' / 'lgbm_gbdt.joblib'),
    ('balanced_rf', CAT_DIR / 'balanced_rf' / 'artifacts' / 'balanced_rf.joblib'),
    ('xgboost', CAT_DIR / 'xgboost' / 'artifacts' / 'xgboost.joblib'),
]

META_MODEL  = CAT_DIR / 'meta' / 'artifacts' / 'meta.joblib'
META_CLASSES = CAT_DIR / 'meta' / 'artifacts' / 'meta_classes.json'
LABEL_MODEL = LABEL_DIR / 'lgbm_fl' / 'artifacts' / 'lgbm_fl.joblib'

def load_models(logger):
    logger.info("Loading models...")
    label = joblib.load(LABEL_MODEL)
    logger.info(f"LABEL: {LABEL_MODEL.name}")
    base = []
    for name, path in BASE_LEARNERS:
        assert path.exists(), f"{name} model not found at {path}"
        m = joblib.load(path)
        base.append((name, m))
        logger.info(f"  {name}: {path.name}")
    meta = joblib.load(META_MODEL)
    with open(META_CLASSES) as f:
        classes = json.load(f)
    logger.info(f"Meta: {META_MODEL.name}  classes={classes}")
    return label, base, meta, classes


def run_label(model, X_label: np.ndarray, threshold: float, logger) -> np.ndarray:
    probs  = model.predict_proba(X_label)[:, 1]
    flagged = probs >= threshold
    logger.info(f"label: flagged={flagged.sum()} / {len(flagged)} ({flagged.mean()*100}%) packets sent to cat")
    return flagged, probs


def run_cat(base_learners, meta, X_cat_flagged: np.ndarray, classes: list, logger) -> np.ndarray:
    probs_list = []
    for name, model in base_learners:
        p = model.predict_proba(X_cat_flagged) # (n_flagged, n_classes)
        probs_list.append(p)
        logger.info(f"{name}: predict_proba shape={p.shape}")

    X_meta = np.hstack(probs_list)

    X_meta = np.clip(X_meta, 1e-7, 1 - 1e-7)
    X_meta = np.log(X_meta / (1 - X_meta))

    preds_idx = meta.predict(X_meta)
    
    return np.array([classes[idx] for idx in preds_idx], dtype=object)


def evaluate(logger):
    label_model, base_learners, meta, classes = load_models(logger)

    X_label, y_label, y_cat = load_test(stage='label')
    X_cat, _, _ = load_test(stage='cat')
    X_label_arr = X_label.values.astype(np.float64)
    X_cat_arr = X_cat.values.astype(np.float64)
    y_true_cat = y_cat.values
    y_true_bin = y_label.values.astype(int)

    n_total  = len(y_true_cat)
    n_attack = int(y_true_bin.sum())
    n_normal = n_total - n_attack
    logger.info(f"Test shape: {X_label.shape}")
    logger.info(f"Total: {n_total}  Attack={n_attack}  Normal={n_normal}")

    flagged, label_probs = run_label(label_model, X_label_arr, LABEL_THRESHOLD, logger)

    label_preds  = flagged.astype(int)
    label_tp = int(((label_preds == 1) & (y_true_bin == 1)).sum())
    label_tn = int(((label_preds == 0) & (y_true_bin == 0)).sum())
    label_fp = int(((label_preds == 1) & (y_true_bin == 0)).sum())
    label_fn = int(((label_preds == 0) & (y_true_bin == 1)).sum())
    logger.info(f"label: TP={label_tp}  FP={label_fp}  TN={label_tn}  FN={label_fn}")
    logger.info(f"label: Recall={label_tp/(label_tp+label_fn+1e-9)} Precision={label_tp/(label_tp+label_fp+1e-9)}")

    X_cat_flagged = X_cat_arr[flagged]
    cat_preds = run_cat(base_learners, meta, X_cat_flagged, classes, logger)
    logger.info(f"cat: classified {len(cat_preds)} packets")

    system_preds = np.full(n_total, 'Normal', dtype=object)
    system_preds[flagged] = cat_preds

    report_str = classification_report(y_true_cat, system_preds, labels=classes, target_names=classes, zero_division=0)
    logger.info(f"\n{report_str}")

    macro_recall = recall_score(y_true_cat, system_preds, average='macro', zero_division=0)
    macro_prec = precision_score(y_true_cat, system_preds, average='macro', zero_division=0)
    macro_f1 = f1_score(y_true_cat, system_preds, average='macro', zero_division=0)
    logger.info(f"Macro Recall: {macro_recall}")
    logger.info(f"Macro Precision: {macro_prec}")
    logger.info(f"Macro F1: {macro_f1}")

    # Per-class breakdown with bar chart
    logger.info("")
    logger.info("Per-class Recall:")
    per_class = {}
    for cls in classes:
        mask  = y_true_cat == cls
        if mask.sum() == 0:
            continue
        r = recall_score(y_true_cat[mask], system_preds[mask], average='micro', zero_division=0)
        p  = precision_score(y_true_cat, system_preds, labels=[cls], average='micro', zero_division=0)
        f1    = f1_score(y_true_cat, system_preds, labels=[cls], average='micro', zero_division=0)
        per_class[cls] = {'recall': float(r), 'precision': float(p), 'f1': float(f1), 'n': int(mask.sum())}
        logger.info(f"{cls}: Recall={r}  Prec={p}  F1={f1} (n={mask.sum()})")

    cm = confusion_matrix(y_true_cat, system_preds, labels=classes)
    fig, ax = plt.subplots(figsize=(13, 11))
    ConfusionMatrixDisplay(cm, display_labels=classes).plot(ax=ax, colorbar=True, cmap='Blues', xticks_rotation=45)
    ax.set_title('System Confusion Matrix - label -> cat')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'confusion_system.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved -> figures/confusion_system.png")

    rs = [per_class[c]['recall'] for c in classes]
    ps = [per_class[c]['precision'] for c in classes]
    x = np.arange(len(classes))
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(x - 0.2, rs, 0.4, label='Recall', color='steelblue')
    ax.bar(x + 0.2, ps, 0.4, label='Precision', color='darkorange', alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(classes, rotation=40, ha='right')
    ax.axhline(macro_recall, color='steelblue', ls='--', lw=1, label=f'Macro Recall={macro_recall}')
    ax.axhline(macro_prec,   color='darkorange', ls='--', lw=1, label=f'Macro Prec={macro_prec}')
    ax.set_ylim(0, 1.05); ax.set_ylabel('Score'); ax.legend()
    ax.set_title('System-level Per-class Recall & Precision')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'per_class_system.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved -> figures/per_class_system.png")

    logger.info(f"Correctly routed as Normal (TN): {label_tn} -> count as correct Normal predictions at system level")
    logger.info(f"Attacks missed by label (FN): {label_fn} -> system-level FN (misclassified as Normal)")
    logger.info(f"Hard Normals passed to cat: {label_fp} -> cat must reclassify these as Normal")
    cat_normal_correct = int((system_preds[flagged] == 'Normal').sum() & (y_true_cat[flagged] == 'Normal').sum())
    logger.info(f"System Normal recall: {per_class['Normal']['recall']} (label TN={label_tn} + cat Normal corrections)")

    probs_list = []
    for name, model in base_learners:
        probs_list.append(model.predict_proba(X_cat_flagged))
    X_meta_probs = np.hstack(probs_list)

    # Extract Shannon Entropy for each base learner's 10-class output profile (Or agony, for short :<)
    from scipy.stats import entropy
    logger.info("Per-Model Prediction Entropy Across Filtered Attack Traffic:")
    for idx, (name, _) in enumerate(base_learners):
        start_col = idx * 10
        end_col = start_col * 10 + 10
        model_probs = X_meta_probs[:, start_col:end_col]
        
        # Calculate row-wise entropy (base 2)
        row_entropies = entropy(model_probs.T, base=2)
        mean_entropy = np.mean(row_entropies)
        max_entropy = np.log2(10) # ~3.3219 bits for 10 classes
        high_entropy_pct = np.mean(row_entropies > 2.0) * 100
        
        logger.info(f"-> {name}: Mean Entropy = {mean_entropy} / {max_entropy} bits ({high_entropy_pct}% of packets exhibit Entropy > 2.0)")

    # Empirical Risk Minimization (ERM) Multi-Class Deflection
    logger.info("Tracking Minority Attack Class Redirection to Dominant Coordinates:")
    results_df = pd.DataFrame({'True': y_true_cat, 'Pred': system_preds})
    
    # Isolate packets that actually passed label filter to audit cat routing behavior
    flagged_results = results_df.iloc[flagged]
    dominant_classes = ['Exploits', 'Generic']
    
    for minority_class in ['Worms', 'Analysis', 'Shellcode']:
        subset = flagged_results[flagged_results['True'] == minority_class]
        total_flagged = len(subset)
        if total_flagged > 0:
            deflected = subset[subset['Pred'].isin(dominant_classes)].shape[0]
            deflection_pct = (deflected / total_flagged) * 100
            logger.info(f"-> Minority Class [{minority_class}]: Real Rows Flagged = {total_flagged} - Deflected into {dominant_classes} = {deflected} ({deflection_pct}%)")

    report = {
        'label': {
            'threshold': LABEL_THRESHOLD,
            'tp': label_tp, 'fp': label_fp, 'tn': label_tn, 'fn': label_fn,
            'recall': label_tp / (label_tp + label_fn + 1e-9),
            'precision': label_tp / (label_tp + label_fp + 1e-9),
            'packets_to_cat': int(flagged.sum())
        },
        'system': {
            'macro_recall': float(macro_recall),
            'macro_precision': float(macro_prec),
            'macro_f1': float(macro_f1),
            'per_class': per_class
        },
    }
    out = REPORT_DIR / 'system_report.json'
    with open(out, 'w') as f:
        json.dump(report, f, indent=2)
    return report


if __name__ == '__main__':
    logger = make_logger(__name__, str(REPORT_DIR / 'eval.log'))
    evaluate(logger)
