import warnings
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import mutual_info_classif, RFECV
from sklearn.metrics import make_scorer, recall_score
from sklearn.inspection import permutation_importance
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
import lightgbm as lgb
import shap
import json
import joblib
import sys

warnings.filterwarnings('ignore')

PATH = Path(__file__).parent.parent
DATASET_PATH = PATH / 'dataset'
OUTPUT_PATH = DATASET_PATH / 'output'
ARTIFACTS_PATH = PATH / 'artifacts'
PL_PATH = ARTIFACTS_PATH / 'preprocessing_pipeline.joblib'
A_PATH = ARTIFACTS_PATH / 'artifacts.json'

for p in [DATASET_PATH, OUTPUT_PATH, ARTIFACTS_PATH]:
    p.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)
fm = logging.Formatter('[%(relativeCreated)d ms] %(levelname)s - %(funcName)s:%(lineno)d - %(message)s')
fh = logging.FileHandler(Path(__file__).parent / 'data_preprocessing.log', mode='w')
ch = logging.StreamHandler(sys.stdout)
fh.setFormatter(fm)
ch.setFormatter(fm)
logger.addHandler(fh)
logger.addHandler(ch)
logger.setLevel(logging.INFO)

DROPS = ['smean', 'dmean', 'sload', 'dload']
LOG1P_TARGETS = ['sbytes', 'dbytes', 'spkts', 'dpkts', 'sloss', 'dloss', 'sinpkt', 'dinpkt', 'sjit', 'djit', 'response_body_len']
FLAGS = ["is_ftp_login", "is_sm_ips_ports", "is_tcp", "is_ftp", "is_http", "tcp_seq_established", "tcp_seq_one_sided", "is_zero_dur", "is_short_flow", "zero_win"]

def audit(df: pd.DataFrame) -> None:
    logger.info(f"Shape: {df.shape}")
    logger.info(f"Null counts (Should be 0): {df.isnull().sum().sum()}")
    logger.info(f"attack_cat classes: {len(df['attack_cat'].unique())}")

def process(df: pd.DataFrame, is_train: bool, artifacts: dict) -> tuple:
    df = df.copy()
    df = df.drop(columns=['id'], errors='ignore')
    y_label = df['label'].copy().reset_index(drop=True)
    y_cat = df['attack_cat'].copy().reset_index(drop=True)
    
    leaks_to_drop = ['label', 'attack_cat', 'proto', 'service', 'state']
    df = df.drop(columns=[col for col in leaks_to_drop if col in df.columns]).reset_index(drop=True)

    df['tcp_seq_established'] = ((df['stcpb'] != 0) & (df['dtcpb'] != 0)).astype(int)
    df['tcp_seq_one_sided'] = ((df['stcpb'] != 0) ^ (df['dtcpb'] != 0)).astype(int)
    df = df.drop(columns=['stcpb', 'dtcpb'], errors='ignore')

    df = df.drop(columns=[col for col in DROPS if col in df.columns], errors='ignore')

    if is_train:
        corr, _ = spearmanr(df['rate'], (df['spkts'] + df['dpkts']) / (df['dur'] + 1e-9))
        is_drop = bool(abs(corr) > 0.95)
        artifacts['is_drop'] = is_drop
        artifacts['corr'] = float(corr)
        logger.info(f"Spearman correlation (rate): {corr} -> Drop: {is_drop}")
    else:
        is_drop = artifacts.get('is_drop', False)
        
    if is_drop and 'rate' in df.columns:
        df = df.drop(columns=['rate'])

    df['bytes_ratio'] = np.log1p(df['sbytes'] / (df['dbytes'] + 1e-9))
    df['pkts_ratio'] = np.log1p(df['spkts'] / (df['dpkts'] + 1e-9))
    df['bytes_per_pkt_src'] = np.log1p(df['sbytes'] / (df['spkts'] + 1e-9))
    df['bytes_per_pkt_dst'] = np.log1p(df['dbytes'] / (df['dpkts'] + 1e-9))
    df['jitter_ratio'] = np.log1p(df['sjit'] / (df['djit'] + 1e-9))
    df['interpacket_ratio'] = np.log1p(df['sinpkt'] / (df['dinpkt'] + 1e-9))
    df['synack_ratio'] = df['synack'] / (df['tcprtt'] + 1e-9)
    df['ack_ratio'] = df['ackdat'] / (df['tcprtt'] + 1e-9)

    for feature in LOG1P_TARGETS:
        if feature in df.columns:
            df[feature] = np.log1p(df[feature])
            
    df['is_zero_dur'] = (df['dur'] == 0).astype(int)
    df['is_short_flow'] = ((df['dur'] > 0) & (df['dur'] < 0.001)).astype(int)
    df['dur'] = np.log1p(df['dur'])

    df['zero_win'] = ((df['swin'] == 0) | (df['dwin'] == 0)).astype(int)
    df['win_asymmetry'] = np.abs(df['swin'] - df['dwin']) / (df['swin'] + df['dwin'] + 1)

    df['srv_diversity'] = 1.0 - np.clip(df['ct_srv_src'] / (df['ct_src_ltm'] + 1e-9), 0.0, 1.0)
    df['dst_concentration'] = np.clip(df['ct_dst_src_ltm'] / (df['ct_src_ltm'] + 1e-9), 0.0, 1.0)

    logger.info("Engineered robust behavioural features, removed protocol dependencies.")
    return df, y_label, y_cat, artifacts

def select(X_train: pd.DataFrame, y_label: pd.Series, y_cat: pd.Series) -> tuple:
    X_train = X_train.astype(float)
    features = X_train.columns.tolist()
    flag_cnt = {f: 0 for f in features}
    flag_info = {f: [] for f in features}

    def flag(feat, method):
        flag_cnt[feat] += 1
        flag_info[feat].append(method)

    low_var = [col for col in features if X_train[col].var() < 0.01]
    logger.info(f"Low variance (<0.01): {low_var if low_var else 'None'}")
    for f in low_var:
        flag(f, 'low_var')

    cont = [c for c in features if c not in FLAGS]
    corr_matrix = X_train[cont].corr(method='spearman').abs()
    high_corr = []
    for i, c1 in enumerate(cont):
        for j, c2 in enumerate(cont):
            if j > i and corr_matrix.loc[c1, c2] > 0.90:
                high_corr.append((c1, c2, corr_matrix.loc[c1, c2]))
    logger.info(f"Highly correlated Spearman pairs (>0.90): {len(high_corr)}")

    vif_features = [c for c in cont if X_train[c].nunique() > 2]
    vif_vals = {}
    vif_arr = X_train[vif_features].values
    for i, col in enumerate(vif_features):
        vif_vals[col] = variance_inflation_factor(vif_arr, i)
    high_vif = {k: v for k, v in vif_vals.items() if v > 10}
    logger.info(f"High VIF (>10): {list(high_vif.keys()) if high_vif else 'None'}")

    mi_label = pd.Series(mutual_info_classif(X_train, y_label, random_state=42), index=features)
    mi_cat = pd.Series(mutual_info_classif(X_train, y_cat, random_state=42), index=features)
    low_mi = set(mi_label[mi_label < mi_label.quantile(0.1)].index) & set(mi_cat[mi_cat < mi_cat.quantile(0.1)].index)
    for f in low_mi:
        flag(f, 'low_mi')

    X_arr = X_train.values
    lgb_label = lgb.LGBMClassifier(n_estimators=300, random_state=42, n_jobs=-1, is_unbalance=True, verbose=-1)
    lgb_label.fit(X_arr, y_label)
    gain_label = pd.Series(lgb_label.booster_.feature_importance(importance_type='gain'), index=features)

    lgb_cat = lgb.LGBMClassifier(n_estimators=300, random_state=42, n_jobs=-1, objective='multiclass', num_class=10, class_weight='balanced', verbose=-1)
    lgb_cat.fit(X_arr, y_cat)
    gain_cat = pd.Series(lgb_cat.booster_.feature_importance(importance_type='gain'), index=features)

    low_gain = set(gain_label[gain_label < gain_label.quantile(0.05)].index) & set(gain_cat[gain_cat < gain_cat.quantile(0.05)].index)
    for f in low_gain:
        flag(f, 'low_gain')

    binary_scorer = make_scorer(recall_score, pos_label=1, zero_division=0)
    perm = permutation_importance(lgb_label, X_arr, y_label, scoring=binary_scorer, n_repeats=5, random_state=42, n_jobs=-1)
    low_perm = set(pd.Series(perm.importances_mean, index=features)[lambda x: x < 0.0001].index)
    for f in low_perm:
        flag(f, 'low_perm')

    idx = np.random.default_rng(42).choice(len(X_arr), min(5000, len(X_arr)), replace=False)
    explainer = shap.TreeExplainer(lgb_cat)
    shap_vals = explainer.shap_values(X_arr[idx])
    shap_mean = np.mean([np.abs(sv).mean(axis=0) for sv in shap_vals], axis=0) if isinstance(shap_vals, list) else np.abs(shap_vals).mean(axis=(0, 2))
    low_shap = set(pd.Series(shap_mean, index=features)[lambda x: x < x.quantile(0.05)].index)
    for f in low_shap:
        flag(f, 'low_shap')

    d = {f for f, c in flag_cnt.items() if c >= 2}
    for c1, c2, _ in high_corr:
        l = c1 if (gain_label[c1] + gain_cat[c1]) < (gain_label[c2] + gain_cat[c2]) else c2
        if flag_cnt[l] >= 1:
            d.add(l)
    for feat in high_vif:
        if flag_cnt[feat] >= 1:
            d.add(feat)
            
    logger.info(f"Dropping {len(d)} weak/collinear features: {sorted(d)}")
    features_after = [f for f in features if f not in d]

    logger.info("Running RFECV Pass 1 (Stage 1 Binary)...")
    sss1 = StratifiedShuffleSplit(n_splits=1, train_size=min(30000, len(X_arr)), random_state=42)
    idx1, _ = next(sss1.split(X_train[features_after], y_label))
    rfecv_label = RFECV(
        estimator=lgb.LGBMClassifier(n_estimators=100, random_state=42, n_jobs=-1, is_unbalance=True, verbose=-1),
        step=1, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring=binary_scorer, min_features_to_select=max(10, len(features_after) // 3), n_jobs=-1
    )
    rfecv_label.fit(X_train[features_after].values[idx1], y_label.values[idx1])
    selected_label = [f for f, sel in zip(features_after, rfecv_label.support_) if sel]

    logger.info("Running RFECV Pass 2 (Stage 2 Multi-class)...")
    sss2 = StratifiedShuffleSplit(n_splits=1, train_size=min(30000, len(X_arr)), random_state=42)
    idx2, _ = next(sss2.split(X_train[features_after], y_cat))
    rfecv_cat = RFECV(
        estimator=lgb.LGBMClassifier(n_estimators=100, random_state=42, n_jobs=-1, objective='multiclass', num_class=10, class_weight='balanced', verbose=-1),
        step=1, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='recall_macro',
        min_features_to_select=max(10, len(features_after) // 3), n_jobs=-1
    )
    rfecv_cat.fit(X_train[features_after].values[idx2], y_cat.values[idx2])
    selected_cat = [f for f, sel in zip(features_after, rfecv_cat.support_) if sel]

    CAT_ALWAYS_KEEP = ['ct_state_ttl']
    for f in CAT_ALWAYS_KEEP:
        if f in features and f not in selected_cat:
            selected_cat.append(f)

    return selected_label, selected_cat

class Pipeline(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler_ = RobustScaler()
        self.cont_cols_ = []
        self.output_cols_ = []

    def fit(self, X: pd.DataFrame, y_label: pd.Series = None, y_cat: pd.Series = None):
        self.cont_cols_ = [c for c in X.columns if c not in FLAGS]
        self.scaler_.fit(X[self.cont_cols_])
        self.output_cols_ = X.columns.tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        X_out[self.cont_cols_] = self.scaler_.transform(X_out[self.cont_cols_])
        return X_out

    def get_stage_cols(self, pre_enc_features: list) -> list:
        return [c for c in self.output_cols_ if c in pre_enc_features]

def f_train(filename):
    df = pd.read_csv(DATASET_PATH / filename)
    audit(df)
    X, y_label, y_cat, artifacts = process(df=df, is_train=True, artifacts={})
    
    selected_label, selected_cat = select(X_train=X, y_label=y_label, y_cat=y_cat)
    artifacts['selected_label'] = selected_label
    artifacts['selected_cat'] = selected_cat

    union = list(dict.fromkeys(selected_label + selected_cat))
    artifacts['selected_union'] = union

    X_union = X[union].copy()
    pl = Pipeline()
    pl.fit(X_union)
    X_t = pl.transform(X_union)

    artifacts['selected_label_cols'] = pl.get_stage_cols(selected_label)
    artifacts['selected_cat_cols'] = pl.get_stage_cols(selected_cat)

    X_t['label'] = y_label.values
    X_t['attack_cat'] = y_cat.values
    X_t.to_parquet(OUTPUT_PATH / 'train.parquet', index=False)
    
    joblib.dump(pl, PL_PATH)
    with open(A_PATH, 'w') as f:
        json.dump(artifacts, f, indent=4)
        
    return {
        'X_train_shape': X_t.shape, 'selected_label_count': len(selected_label),
        'selected_cat_count': len(selected_cat), 'union_count': len(union)
    }

def f_test(filename):
    assert PL_PATH.exists() and A_PATH.exists(), "Run f_train first!"
    pl = joblib.load(PL_PATH)
    with open(A_PATH) as f:
        artifacts = json.load(f)
        
    df = pd.read_csv(DATASET_PATH / filename)
    audit(df)
    X, y_label, y_cat, _ = process(df=df, is_train=False, artifacts=artifacts)
    
    union = artifacts['selected_union']
    missing = [c for c in union if c not in X.columns]
    assert not missing, f"Missing features in test set: {missing}"
    
    X_t = pl.transform(X[union].copy())
    X_t['label'] = y_label.values
    X_t['attack_cat'] = y_cat.values
    X_t.to_parquet(OUTPUT_PATH / 'test.parquet', index=False)
    return {'X_test_shape': X_t.shape}

def main():
    logger.info("Starting Data Preprocessing Pipeline...")
    train_res = f_train('trainset.csv')
    test_res = f_test('testset.csv')
    logger.info(f"TRAIN COMPLETED: {train_res}")
    logger.info(f"TEST COMPLETED: {test_res}")

if __name__ == "__main__":
    main()
