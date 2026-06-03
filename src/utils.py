import logging
import json
import sys
from pathlib import Path
import pandas as pd

PATH = Path(__file__).parent
DATA_DIR = PATH / 'dataset' / 'output'
ARTIFACTS_DIR = PATH / 'artifacts'

def make_logger(name: str, f: str) -> logging.Logger:
    l = logging.getLogger(name)
    fm = logging.Formatter('[%(relativeCreated)d ms] %(levelname)s - %(funcName)s:%(lineno)d - %(message)s')
    fh = logging.FileHandler(f, mode='w')
    ch = logging.StreamHandler(sys.stdout)
    fh.setFormatter(fm)
    ch.setFormatter(fm)
    l.addHandler(fh)
    l.addHandler(ch)
    l.setLevel(logging.INFO)
    return l

def _load(filename, stage: str) -> tuple:
    assert stage in ('label', 'cat'), f"Stage must be 'label' or 'cat', got '{stage}'"
    df = pd.read_parquet(DATA_DIR / filename)
    y_label = df['label'].reset_index(drop=True)
    y_cat = df['attack_cat'].reset_index(drop=True)
    with open(ARTIFACTS_DIR / 'artifacts.json') as f:
        artifacts = json.load(f)
    X = df[artifacts[f'selected_{stage}_cols']].reset_index(drop=True)
    return X, y_label, y_cat

def load_train(stage: str) -> tuple:
    return _load('train.parquet', stage)

def load_test(stage: str) -> tuple:
    return _load('test.parquet', stage)
