from pathlib import Path

import pytest

from amia import (
    DEFAULT_IMG_SIZE_CSV,
    DEFAULT_IMAGE_DICT_PATH,
    DEFAULT_TRAIN_CSV,
    DEFAULT_TRAIN_DIR,
)
from optuna_search import run_study


def require_dataset():
    missing = []
    if not DEFAULT_TRAIN_DIR.exists():
        missing.append(str(DEFAULT_TRAIN_DIR))
    if not DEFAULT_IMAGE_DICT_PATH.exists():
        if not DEFAULT_TRAIN_CSV.exists():
            missing.append(str(DEFAULT_TRAIN_CSV))
        if not DEFAULT_IMG_SIZE_CSV.exists():
            missing.append(str(DEFAULT_IMG_SIZE_CSV))
    image_files = list(Path(DEFAULT_TRAIN_DIR).glob("*.png"))
    if not image_files:
        missing.append(f"{DEFAULT_TRAIN_DIR} (no .png files)")
    if missing:
        pytest.skip(f"Dataset not available for smoke test: {', '.join(missing)}")


def test_optuna_smoke_fasterrcnn(tmp_path: Path):
    require_dataset()
    storage_path = tmp_path / "optuna_fasterrcnn.db"
    study = run_study(
        model_type="fasterrcnn",
        trials=1,
        storage_path=storage_path,
        seed=123420,
        train_limit=2,
        val_limit=2,
    )
    assert len(study.trials) >= 1
