import warnings
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, RobustScaler, OneHotEncoder
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
fh = logging.FileHandler('data_preprocessing.log')
ch = logging.StreamHandler(sys.stdout)
fh.setFormatter(fm)
ch.setFormatter(fm)
logger.addHandler(fh)
logger.addHandler(ch)
logger.setLevel(logging.INFO)

DROPS = ['smean', 'dmean', 'sload', 'dload']
LOG1P = ['sbytes', 'dbytes', 'spkts', 'dpkts', 'sloss', 'dloss', 'sinpkt', 'dinpkt', 'sjit', 'djit', 'response_body_len']
FLAGS = ["is_ftp_login", "is_sm_ips_ports", "is_tcp", "is_ftp", "is_http", "tcp_seq_established", "tcp_seq_one_sided", "is_zero_dur", "is_short_flow", "zero_win"]

def audit(df: pd.DataFrame) -> None:
    logger.info(f"Shape: {df.shape}")
    logger.info(f"Null counts (Should be 0): {df.isnull().sum().sum()}")
    logger.info(f"attack_cat: {len(df['attack_cat'].unique())}")

def process(df: pd.DataFrame, is_train: bool, artifacts: dict) -> tuple:
    df = df.drop(columns=['id'])
    y_label = df['label'].copy().reset_index(drop=True)
    y_cat = df['attack_cat'].copy().reset_index(drop=True)
    df = df.drop(columns=['label', 'attack_cat']).reset_index(drop=True)
    
    df['tcp_seq_established'] = ((df['stcpb'] != 0) & (df['dtcpb'] != 0)).astype(int)
    df['tcp_seq_one_sided'] = ((df['stcpb'] != 0) ^ (df['dtcpb'] != 0)).astype(int)
    df = df.drop(columns=['stcpb', 'dtcpb'])

    df = df.drop(columns=DROPS)

    if is_train:
        corr, _ = spearmanr(df['rate'], (df['spkts'] + df['dpkts']) / (df['dur'] + 1e-9))
        is_drop = bool(abs(corr) > 0.95) # Change? (90% should be good)
        artifacts['is_drop'] = is_drop
        artifacts['corr'] = round(float(corr), 6)
        logger.info(f"Spearman correlation: {corr:.6f} -> Drop: {is_drop}")
    else:
        is_drop = artifacts['is_drop']
    if is_drop:
        df = df.drop(columns=['rate'])

    df['is_tcp'] = (df['proto'] == 'tcp').astype(int)
    df['is_ftp'] = df['service'].isin(['ftp', 'ftp-data']).astype(int)
    df['is_http'] = df['service'].isin(['http', 'ssl']).astype(int)
    
    for feature in LOG1P:
        df[feature] = np.log1p(df[feature])
    df['is_zero_dur'] = (df['dur'] == 0).astype(int)
    df['is_short'] = ((df['dur'] > 0) & (df['dur'] < 0.001)).astype(int)
    df['dur'] = np.log1p(df['dur'])

    df['bytes_ratio'] = df['sbytes'] - df['dbytes']
    df['pkts_ratio'] = df['spkts'] - df['dpkts']
    df['bytes_per_pkt_src'] = df['sbytes'] - df['spkts']
    df['bytes_per_pkt_dst'] = df['dbytes'] - df['dpkts']
    df['jitter_ratio'] = df['sjit'] - df['djit']
    df['interpacket_ratio'] = df['sinpkt'] - df['dinpkt']
    df['synack_ratio'] = df['synack'] / (df['tcprtt'] + 1e-9)
    df['ack_ratio'] = df['ackdat'] / (df['tcprtt'] + 1e-9)

    df['zero_win'] = ((df['swin'] == 0) | (df['dwin'] == 0)).astype(int)
    df['win_asymmetry'] = np.abs(df['swin'] - df['dwin']) / (df['swin'] + df['dwin'] + 1)
    
    return df, y_label, y_cat, artifacts

def select(X_train: pd.DataFrame, y_label: pd.Series, y_cat: pd.Series) -> list:
    for col in ['proto', 'service', 'state']:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col].astype(str))
    X_train = X_train.astype(float)
    features = X_train.columns.tolist()
    flag_cnt = {f: 0 for f in features}
    flag_info = {f: [] for f in features}
    def flag(feat, method):
        flag_cnt[feat] += 1
        flag_info[feat].append(method)

    low_var = []
    for col in features:
        if X_train[col].var() < 0.01:
            low_var.append(col)
            flag(col, 'low_var')
    logger.info(f"Low variance (<0.01): {low_var if low_var else 'NO!'}")

    cont = [c for c in features if c not in FLAGS]
    corr_matrix = X_train[cont].corr(method='spearman').abs()
    high_corr = []
    seen = set()
    for i, c1 in enumerate(cont):
        for j, c2 in enumerate(cont):
            if j <= i:
                continue
            if corr_matrix.loc[c1, c2] > 0.90:
                high_corr.append((c1, c2, corr_matrix.loc[c1, c2]))
                seen.add((c1, c2))
    logger.info(f"Spearman pairs (>0.90): {len(high_corr)}") # 0.90 LESS!!!
    for c1, c2, r in high_corr:
        logger.info(f"{c1} - {c2}: {r:.6f}")

    vif_features = [c for c in cont if X_train[c].nunique() > 2]
    vif_vals = {}
    vif_arr = X_train[vif_features].values.astype(float)
    for i, col in enumerate(vif_features):
        vif_vals[col] = variance_inflation_factor(vif_arr, i)
    high_vif = {k: v for k, v in vif_vals.items() if v > 10}
    logger.info(f"High VIF (>10): {list(high_vif.keys()) if high_vif else 'NO!'}")

    mi_label = pd.Series(mutual_info_classif(X_train, y_label, random_state=42), index=features)
    mi_cat = pd.Series(mutual_info_classif(X_train, y_cat, random_state=42), index=features)
    thresh_label = mi_label.quantile(0.1)
    thresh_cat = mi_cat.quantile(0.1)
    low_mi = set(mi_label[mi_label < thresh_label].index) & set(mi_cat[mi_cat < thresh_cat].index)
    logger.info(f"Low MI (0.1): {list(low_mi) if low_mi else 'NO!'}")
    for f in low_mi:
        flag(f, 'low_mi')

    X_arr = X_train.values
    lgb_label = lgb.LGBMClassifier(n_estimators=300, random_state=42, n_jobs=-1, is_unbalance=True, verbose=-1)
    lgb_label.fit(X_arr, y_label)
    gain_label = pd.Series(lgb_label.booster_.feature_importance(importance_type='gain'), index=features)
    lgb_cat = lgb.LGBMClassifier(n_estimators=300, random_state=42, n_jobs=-1, objective='multiclass', num_class=10, class_weight='balanced', verbose=-1)
    lgb_cat.fit(X_arr, y_cat)
    gain_cat = pd.Series(lgb_cat.booster_.feature_importance(importance_type='gain'), index=features)
    # I prefer 0.05 :>
    thresh_glabel = gain_label.quantile(0.05)
    thresh_gcat = gain_cat.quantile(0.05)
    low_gain = set(gain_label[gain_label < thresh_glabel].index) & set(gain_cat[gain_cat < thresh_gcat].index)
    logger.info(f"Low gain (0.05): {list(low_gain) if low_gain else 'NO!'}")
    for f in low_gain:
        flag(f, 'low_gain')

    mr_scorer = make_scorer(recall_score, average='macro', zero_division=0, pos_label=None)
    perm = permutation_importance(lgb_cat, X_arr, y_cat, scoring=mr_scorer, n_repeats=5, random_state=42, n_jobs=-1)
    perm_imp = pd.Series(perm.importances_mean, index=features)
    low_perm = set(perm_imp[perm_imp < 0.0001].index) # Google for 0.0001 :>
    logger.info(f"Near-zero permutation importance: {list(low_perm) if low_perm else 'NO!'}")
    for f in low_perm:
        flag(f, 'low_perm')
    
    idx = np.random.default_rng(42).choice(len(X_arr), min(5000, len(X_arr)), replace=False)
    explainer = shap.TreeExplainer(lgb_cat)
    shap_vals = explainer.shap_values(X_arr[idx])
    if isinstance(shap_vals, list):
        shap_mean = np.mean([np.abs(sv).mean(axis=0) for sv in shap_vals], axis=0)
    else:
        shap_mean = np.abs(shap_vals).mean(axis=(0, 2))
    shap_imp = pd.Series(shap_mean, index=features)
    low_shap = set(shap_imp[shap_imp < shap_imp.quantile(0.05)].index)
    logger.info(f"Low SHAP (0.05): {list(low_shap) if low_shap else 'NO!'}")
    for f in low_shap:
        flag(f, 'low_shap')

    d = {f for f, c in flag_cnt.items() if c >= 2}
    for c1, c2, _ in high_corr:
        i_c1 = gain_label[c1] + gain_cat[c1]
        i_c2 = gain_label[c2] + gain_cat[c2]
        l = c1 if i_c1 < i_c2 else c2
        if flag_cnt[l] >= 1:
            d.add(l)
            flag(l, 'high_corr_lower_imp')
    for feat in high_vif:
        if flag_cnt[feat] >= 1:
            d.add(feat)
            flag(feat, 'high_vif')
    logger.info(f"Dropping {len(d)} features: {sorted(d)}")
    features_after = [f for f in features if f not in d]

    sss = StratifiedShuffleSplit(n_splits=1, train_size=min(30000, len(X_arr)), random_state=42)
    sample_train_idx, _ = next(sss.split(X_train[features_after], y_cat))
    X_rfecv = X_train[features_after].values[sample_train_idx]
    y_rfecv = y_cat.values[sample_train_idx]
    rfecv = RFECV(estimator=lgb.LGBMClassifier(n_estimators=100, random_state=42, n_jobs=-1, objective='multiclass', num_class=10, class_weight='balanced', verbose=-1), step=1, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), scoring=make_scorer(recall_score, average='macro',zero_division=0, pos_label=None), min_features_to_select=max(10, len(features_after) // 3), n_jobs=-1)
    rfecv.fit(X_rfecv, y_rfecv)
    selected_mask = rfecv.support_
    fs = [f for f, sel in zip(features_after, selected_mask) if sel]
    rm = [f for f, sel in zip(features_after, selected_mask) if not sel]
    total = set(features) - set(fs)
    logger.info(f"RFECV count: {rfecv.n_features_}")
    logger.info(f"Removed: {rm}")
    logger.info(f"Before: {len(features)}")
    logger.info(f"Dropped: {len(total)} -> {total}")
    logger.info(f"After: {len(fs)}")
    logger.info(f"Features: {fs}")
    return fs

class SmoothedTargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, column: str, smoothing: float, is_multiclass: bool):
        self.column = column
        self.smoothing = smoothing
        self.is_multiclass = is_multiclass
        self.encoding_maps_: dict = {}
        self.global_means_: dict = {}
        self.classes_: list = []

    def _fit_single(self, col: pd.Series, target: np.ndarray, key):
        global_mean = float(target.mean())
        self.global_means_[key] = global_mean
        tmp = pd.DataFrame({'val': col.values, 't': target})
        enc_map = {}
        for val, grp in tmp.groupby('val'):
            n = len(grp)
            cat_mean = float(grp['t'].mean())
            enc_map[val] = (n * cat_mean + self.smoothing * global_mean) / (n + self.smoothing)
        self.encoding_maps_[key] = enc_map

    def fit(self, X: pd.DataFrame, y: pd.Series):
        col = X[self.column]
        y_arr = np.array(y)
        if self.is_multiclass:
            self.classes_ = sorted(np.unique(y_arr).tolist())
            for cl in self.classes_:
                b = (y_arr == cl).astype(float)
                self._fit_single(col, b, key=cl)
        else:
            b = y_arr.astype(float)
            self._fit_single(col, b, key='binary')
            self.classes_ = ['binary']
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        col = X[self.column]
        res = {}
        for key in self.classes_:
            enc_map = self.encoding_maps_[key]
            global_mean = self.global_means_[key]
            encoded = col.map(enc_map).fillna(global_mean)
            col_name = f"proto_enc_{key}" if self.is_multiclass else f"proto_enc"
            res[col_name] = encoded.values
        return pd.DataFrame(res, index=X.index)

    def get_names(self):
        if self.is_multiclass:
            return [f"proto_enc_{cl}" for cl in self.classes_]
        return ['proto_enc']

class Pipeline(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.proto_enc_label_: SmoothedTargetEncoder | None = None
        self.proto_enc_cat_: SmoothedTargetEncoder | None = None
        self.ohe_: OneHotEncoder | None = None
        self.scaler_: RobustScaler | None = None
        self.ohe_cols_: list = []
        self.cont_cols_: list = []
        self.output_cols_: list = []

    def fit(self, X: pd.DataFrame, y_label: pd.Series, y_cat: pd.Series):
        if 'proto' in X.columns:
            self.proto_enc_label_ = SmoothedTargetEncoder('proto', smoothing=10, is_multiclass=False)
            self.proto_enc_label_.fit(X[['proto']], y_label)
            self.proto_enc_cat_ = SmoothedTargetEncoder('proto', smoothing=10, is_multiclass=True)
            self.proto_enc_cat_.fit(X[['proto']], y_cat)
        self.ohe_cols_ = [c for c in ['service', 'state'] if c in X.columns]
        if self.ohe_cols_:
            self.ohe_ = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            self.ohe_.fit(X[self.ohe_cols_])
        X_encoded = self._apply_encodings(X)
        ohe_names = (self.ohe_.get_feature_names_out(self.ohe_cols_).tolist() if self.ohe_ is not None else [])
        proto_names = (self.proto_enc_label_.get_names() + self.proto_enc_cat_.get_names() if self.proto_enc_label_ is not None else [])
        non_scale_cols = set(FLAGS) | set(ohe_names)
        self.cont_cols_ = [c for c in X_encoded.columns if c not in non_scale_cols]
        self.scaler_ = RobustScaler()
        self.scaler_.fit(X_encoded[self.cont_cols_])
        self.output_cols_ = X_encoded.columns.tolist()
        return self
    
    def _apply_encodings(self, X: pd.DataFrame) -> pd.DataFrame:
        if 'proto' in X.columns and self.proto_enc_label_ is not None and self.proto_enc_cat_ is not None:
            proto_label = self.proto_enc_label_.transform(X[['proto']])
            proto_cat = self.proto_enc_cat_.transform(X[['proto']])
            X = pd.concat([X.drop(columns=['proto']), proto_label, proto_cat], axis=1)
        if self.ohe_cols_ and self.ohe_ is not None:
            ohe_names = self.ohe_.get_feature_names_out(self.ohe_cols_).tolist()
            ohe_df = pd.DataFrame(self.ohe_.transform(X[self.ohe_cols_]), columns=ohe_names, index=X.index)
            X = pd.concat([X.drop(columns=self.ohe_cols_), ohe_df], axis=1)
        return X

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_encoded = self._apply_encodings(X)
        X_encoded[self.cont_cols_] = self.scaler_.transform(X_encoded[self.cont_cols_])
        return X_encoded

def f_train(filename):
    df = pd.read_csv(DATASET_PATH / filename)
    audit(df)
    X, y_label, y_cat, artifacts = process(df=df, is_train=True, artifacts={})
    selected = select(X_train=X, y_label=y_label, y_cat=y_cat)
    artifacts['selected'] = selected
    X_sel = X[selected].copy()
    pl = Pipeline()
    pl.fit(X_sel, y_label, y_cat)
    X_t = pl.transform(X_sel)
    X_t['label'] = y_label.values
    X_t['attack_cat'] = y_cat.values
    X_t.to_parquet(OUTPUT_PATH / 'train.parquet', index=False)
    joblib.dump(pl, PL_PATH)
    with open(A_PATH, 'w') as f:
        json.dump(artifacts, f, indent=2)
    return {'X_train': X_t, 'y_label': y_label, 'y_cat': y_cat, 'pipeline': pl, 'artifacts': artifacts, 'selected': selected}

def f_test(filename):
    assert PL_PATH.exists() and A_PATH.exists(), ":<"
    pl = joblib.load(PL_PATH)
    with open(A_PATH) as f:
        artifacts = json.load(f)
    df = pd.read_csv(DATASET_PATH / filename)
    audit(df)
    X, y_label, y_cat, _ = process(df=df, is_train=False, artifacts=artifacts)
    selected = artifacts['selected']
    assert not [f for f in selected if f not in X.columns], "We might be missing some values..."
    X_sel = X[selected].copy()
    X_t = pl.transform(X_sel)
    X_t['label'] = y_label.values
    X_t['attack_cat'] = y_cat.values
    X_t.to_parquet(OUTPUT_PATH / 'test.parquet', index=False)
    return {'X_test': X_t, 'y_label': y_label, 'y_cat': y_cat}

def main():
    train_res = f_train('trainset.csv')
    test_res = f_test('testset.csv')
    logger.info(f"TRAIN RESULTS:\n{train_res}")
    logger.info(f"TEST RESULTS:\n{test_res}")

if __name__ == "__main__":
    main()
