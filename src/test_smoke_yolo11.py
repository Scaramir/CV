from pathlib import Path

import pytest

from amia import (
    DEFAULT_IMG_SIZE_CSV,
    DEFAULT_IMAGE_DICT_PATH,
    DEFAULT_TRAIN_CSV,
    DEFAULT_TRAIN_DIR,
    TrainingConfig,
    run_training,
)


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


def test_smoke_yolo11():
    pytest.importorskip("ultralytics")
    require_dataset()
    config = TrainingConfig(
        model_type="yolo11",
        epochs=1,
        batch_size=1,
        train_limit=2,
        val_limit=2,
        log_with_mlflow=False,
        use_amp=False,
        auto_scale_batch_size=False,
        yolo_auto_batch=False,
        yolo_imgsz=320,
        yolo_rebuild_dataset=True,
        yolo_mosaic=0.0,
        yolo_mixup=0.0,
        yolo_copy_paste=0.0,
        yolo_hsv_h=0.0,
        yolo_hsv_s=0.0,
        yolo_hsv_v=0.0,
    )
    run_training(config)
