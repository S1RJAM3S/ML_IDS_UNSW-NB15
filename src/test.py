from utils import make_logger, load_train, load_test
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, recall_score
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from pathlib import Path

PATH = Path(__file__).parent
logger = make_logger(__name__, str(PATH / 'theory.log'))

X_train, _, y_train = load_train(stage='cat')
X_test, _, y_test = load_test(stage='cat')

# TEST 1
X_adv = pd.concat([X_train, X_test], axis=0)
y_adv = np.array([0] * len(X_train) + [1] * len(X_test))

X_a_train, X_a_test, y_a_train, y_a_test = train_test_split(X_adv, y_adv, test_size=0.3, random_state=42, stratify=y_adv)

adv_clf = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
adv_clf.fit(X_a_train, y_a_train)

preds = adv_clf.predict_proba(X_a_test)[:, 1]
auc_score = roc_auc_score(y_a_test, preds)

logger.info(f"Adversarial Train vs Test ROC-AUC: {auc_score}")

if auc_score > 0.85:
    logger.info("Kill me")

# TEST 2
def audit_class_collapse(X_train, y_train, X_test, y_test, target_class):
    train_subset = X_train[y_train == target_class]
    test_subset = X_test[y_test == target_class]
    
    if len(train_subset) == 0 or len(test_subset) == 0:
        logger.info(f"Skipping {target_class}: Insufficient data.")
        return None
        
    X_adv = pd.concat([train_subset, test_subset], axis=0)
    y_adv = np.array([0] * len(train_subset) + [1] * len(test_subset))
    
    X_a_train, X_a_test, y_a_train, y_a_test = train_test_split(X_adv, y_adv, test_size=0.3, random_state=42, stratify=y_adv)
    
    clf = lgb.LGBMClassifier(n_estimators=50, random_state=42, verbose=-1)
    clf.fit(X_a_train, y_a_train)
    
    preds = clf.predict_proba(X_a_test)[:, 1]
    return roc_auc_score(y_a_test, preds)

classes_to_test = ['Analysis', 'Generic', 'Worms', 'Exploits']
logger.info("CLASS-SPECIFIC ADVERSARIAL VALIDATION")
for cls in classes_to_test:
    auc = audit_class_collapse(X_train, y_train, X_test, y_test, cls)
    if auc is not None:
        logger.info(f"Class [{cls}] -> Train vs Test Adversarial AUC: {auc}")

def prove_structural_overlap(X_train, y_train, X_test, y_test):
    logger.info("OVERLAP AUDIT")
    
    test_analysis = X_test[y_test == 'Analysis']
    
    for majority_class in ['Generic', 'Exploits']:
        train_majority = X_train[y_train == majority_class]
        
        X_mix = pd.concat([train_majority, test_analysis], axis=0)
        y_mix = np.array([0] * len(train_majority) + [1] * len(test_analysis))
        
        X_m_train, X_m_test, y_m_train, y_m_test = train_test_split(X_mix, y_mix, test_size=0.3, random_state=42, stratify=y_mix)
        
        clf = lgb.LGBMClassifier(n_estimators=50, random_state=42, verbose=-1)
        clf.fit(X_m_train, y_m_train)
        
        preds = clf.predict_proba(X_m_test)[:, 1]
        auc = roc_auc_score(y_m_test, preds)
        
        logger.info(f"Can we separate Test [Analysis] from Train [{majority_class}]? ROC-AUC: {auc}")

prove_structural_overlap(X_train, y_train, X_test, y_test)

def prove_extrapolation_blindspot(X_train, y_train, X_test, y_test, target_class='Analysis'):
    logger.info(f"STRUCTURAL BOUNDARY PROOF FOR [{target_class}]")
    
    train_class = X_train[y_train == target_class]
    test_class = X_test[y_test == target_class]
    
    out_of_bounds_count = 0
    total_features_checked = 0
    
    numerical_cols = X_train.select_dtypes(include=[np.number]).columns
    
    logger.info(f"Auditing {len(numerical_cols)} numerical feature boundaries...")
    
    for col in numerical_cols:
        train_min = train_class[col].min()
        train_max = train_class[col].max()
        
        test_values = test_class[col].values
        violated_rows = (test_values < train_min) | (test_values > train_max)
        violated_pct = np.mean(violated_rows) * 100
        
        if violated_pct > 30: # If more than 30% of test points are out-of-bounds
            logger.info(f"Feature [{col}] - Train Range: [{train_min}, {train_max}]")
            logger.info(f"{violated_pct}% of Real-World Test packets fall OUTSIDE this box!")
            out_of_bounds_count += 1
            
    logger.info(f"{out_of_bounds_count} key features are completely out-of-bounds.")

prove_extrapolation_blindspot(X_train, y_train, X_test, y_test)

def extract_hard_numerical_proof(X_train, y_train, X_test, y_test):
    logger.info("DATA FORENSIC AUDIT: MULTI-VARIABLE INTERSECTION PROOF")
    
    clf = lgb.LGBMClassifier(n_estimators=30, random_state=42, verbose=-1)
    clf.fit(X_train, y_train)
    
    importances = clf.feature_importances_
    top_indices = np.argsort(importances)[::-1][:3]
    top_features = X_train.columns[top_indices].tolist()
    
    logger.info(f"Top 3 decision-making features identified by model: {top_features}\n")
    
    for col in top_features:
        logger.info(f"FEATURE METRIC: {col}")
        
        train_profile = X_train[y_train == 'Analysis'][col].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_dict()
        test_profile = X_test[y_test == 'Analysis'][col].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_dict()
        
        hijack_profile = X_train[y_train == 'Generic'][col].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_dict()
        
        df_metrics = pd.DataFrame({
            'Train [Analysis] (What it learned)': train_profile,
            'Test [Analysis] (What it saw)': test_profile,
            'Train [Generic] (Where it went)': hijack_profile
        })
        logger.info(df_metrics.round(4))

extract_hard_numerical_proof(X_train, y_train, X_test, y_test)

def audit_all_collapsing_classes(X_train, y_train, X_test, y_test):
    logger.info("GLOBAL DATA FORENSIC AUDIT: MINORITY SIGNATURE HIJACKING")
    
    clf = lgb.LGBMClassifier(n_estimators=50, random_state=42, verbose=-1)
    clf.fit(X_train, y_train)
    
    importances = clf.feature_importances_
    sorted_features = [X_train.columns[i] for i in np.argsort(importances)[::-1]]
    
    failing_classes = {
        'Backdoor': 'Generic',
        'DoS': 'Exploits',
        'Worms': 'Generic'
    }
    
    test_features = [f for f in sorted_features if f in ['dur', 'sbytes', 'dbytes', 'sttl', 'dttl']][:2]
    
    for target_class, hijack_class in failing_classes.items():
        logger.info(f"AUDITING TARGET COLLAPSE: [{target_class}] -> Blended into [{hijack_class}]")
        
        for col in test_features:
            logger.info(f"Feature Focus: {col}")
            
            train_profile = X_train[y_train == target_class][col].quantile([0.25, 0.50, 0.75]).to_dict()
            test_profile = X_test[y_test == target_class][col].quantile([0.25, 0.50, 0.75]).to_dict()
            hijack_profile = X_train[y_train == hijack_class][col].quantile([0.25, 0.50, 0.75]).to_dict()
            
            df_metrics = pd.DataFrame({
                f'Train [{target_class}]': train_profile,
                f'Test [{target_class}]': test_profile,
                f'Train [{hijack_class}]': hijack_profile
            })
            logger.info(df_metrics.round(4))

audit_all_collapsing_classes(X_train, y_train, X_test, y_test)

def prove_dataset_artifact_leakage(X_train, y_train, X_test, y_test):
    logger.info("LAB INFRASTRUCTURE LEAKAGE AUDIT - ANALYSIS ATTACK")
    
    infrastructure_features = [f for f in ['sttl', 'dttl', 'swin', 'dwin'] if f in X_train.columns]
    
    if not infrastructure_features:
        logger.info("Required structural features (sttl, swin, etc.) not found in preprocessed DataFrame.")
        return

    for col in infrastructure_features:
        logger.info(f"NETWORK INFRASTRUCTURE ARTIFACT: {col}")
        
        # What did the lab generator hardcode during training?
        train_normal = X_train[y_train == 'Normal'][col].value_counts(normalize=True).head(2).to_dict()
        train_attack = X_train[y_train != 'Normal'][col].value_counts(normalize=True).head(2).to_dict()
        
        # What happened when they captured the test set chronologically later?
        test_class_fail = X_test[y_test == 'Analysis'][col].value_counts(normalize=True).head(2).to_dict()
        
        logger.info(f"Train [Normal Background] top values: {train_normal}")
        logger.info(f"Train [Synthetic Attacks] top values: {train_attack}")
        logger.info(f"Test [Failing Analysis] top values: {test_class_fail}")

prove_dataset_artifact_leakage(X_train, y_train, X_test, y_test)


def prove_total_structural_divergence(X_train, y_train, X_test, y_test, target_class='Analysis'):
    logger.info(f"STRUCTURAL ALIGNMENT AUDIT FOR [{target_class}]")
    
    # Strip away all environmental infrastructure leakage features
    infrastructure_cols = ['sttl', 'dttl', 'swin', 'dwin']
    clean_cols = [c for c in X_train.columns if c not in infrastructure_cols]
    
    # Extract the raw behavioral features (packet sizes, rates, counts)
    train_signals = X_train[y_train == target_class][clean_cols]
    test_signals = X_test[y_test == target_class][clean_cols]
    
    logger.info(f"Analyzing pure behavioral feature space dimensions: {len(clean_cols)} columns")
    logger.info(f"Train samples: {len(train_signals)} - Test samples: {len(test_signals)}")
    
    # Train an Isolation Forest ONLY on the Train footlogger.info to learn its "shape"
    iso = IsolationForest(contamination=0.05, random_state=42)
    iso.fit(train_signals)
    
    # Generate the raw structural anomaly scores (lower = more divergent/alien)
    train_scores = iso.score_samples(train_signals)
    test_scores = iso.score_samples(test_signals)
    
    # Measure the mathematical drift between the distributions
    logger.info("STRUCTURAL SHAPE METRICS (ANOMALY SCORE QUANTILES)")
    logger.info(f"Train [{target_class}] Internal Space: Median = {np.median(train_scores)}, 10th Pct = {np.percentile(train_scores, 10)}")
    logger.info(f"Test  [{target_class}] Internal Space: Median = {np.median(test_scores)}, 10th Pct = {np.percentile(test_scores, 10)}")
    
    anomaly_threshold = np.percentile(train_scores, 5)
    alien_records_pct = np.mean(test_scores < anomaly_threshold) * 100
    
    logger.info(f"{alien_records_pct}% of the Test [{target_class}] records fall entirely outside the multi-dimensional shape learned from the training set.")

prove_total_structural_divergence(X_train, y_train, X_test, y_test, target_class='Analysis')

# I'm too lazy to make another function :<
# Strip the infrastructure proxies
clean_cols = [c for c in X_train.columns if c not in ['sttl', 'dttl', 'swin', 'dwin']]

# Scale the data
scaler = StandardScaler()
X_tr_scaled = scaler.fit_transform(X_train[clean_cols])
X_te_scaled = scaler.transform(X_test[clean_cols])

# Test LightGBM (Axis-Aligned Tree splits)
lgb_model = lgb.LGBMClassifier(n_estimators=50, random_state=42, verbose=-1)
lgb_model.fit(X_train[clean_cols], y_train)
lgb_preds = lgb_model.predict(X_test[clean_cols])

# Test Ridge Classifier (Continuous Hyperplane splits)
linear_model = RidgeClassifier(random_state=42)
linear_model.fit(X_tr_scaled, y_train)
linear_preds = linear_model.predict(X_te_scaled)

logger.info(f"LightGBM [Analysis] Recall: {recall_score(y_test, lgb_preds, average=None, labels=['Analysis'])[0]}")
logger.info(f"Linear [Analysis] Recall: {recall_score(y_test, linear_preds, average=None, labels=['Analysis'])[0]}")
