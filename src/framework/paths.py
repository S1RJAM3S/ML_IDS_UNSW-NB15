from pathlib import Path

ROOT         = Path(__file__).parent.parent.parent
DATASET_DIR  = ROOT / 'dataset' / 'output'
TRAIN_PATH   = DATASET_DIR / 'train.parquet'
TEST_PATH    = DATASET_DIR / 'test.parquet'
ARTIFACTS    = ROOT / 'artifact'
MODELS_DIR   = ROOT / 'src' / 'models'
LABEL_DIR   = MODELS_DIR / 'label'
CAT_DIR   = MODELS_DIR / 'attack_cat'
 
def model_artifacts(stage_dir: Path, model_name: str) -> Path:
    p = stage_dir / model_name / 'artifacts'
    p.mkdir(parents=True, exist_ok=True)
    return p
