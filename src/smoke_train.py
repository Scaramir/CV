import argparse

from amia import TrainingConfig, run_training


def run_smoke_training(num_epochs=2, model="all"):
    configs = []
    if model in ("fasterrcnn", "all"):
        configs.append(
            TrainingConfig(
                model_type="fasterrcnn",
                epochs=num_epochs,
                batch_size=1,
                experiment_name="amia-smoke",
                run_name="fasterrcnn-smoke",
                train_limit=100,
                val_limit=100,
            )
        )
    if model in ("retinanet", "all"):
        configs.append(
            TrainingConfig(
                model_type="retinanet",
                epochs=num_epochs,
                batch_size=1,
                experiment_name="amia-smoke",
                run_name="retinanet-smoke",
                train_limit=100,
                val_limit=100,
            )
        )
    if model in ("yolo11", "all"):
        configs.append(
            TrainingConfig(
                model_type="yolo11",
                epochs=num_epochs,
                batch_size=1,
                experiment_name="amia-smoke",
                run_name="yolo11-smoke",
                train_limit=100,
                val_limit=100,
                yolo_imgsz=320,
                yolo_model="yolo11s.pt",
                yolo_rebuild_dataset=True,
                yolo_mosaic=0.0,
                yolo_mixup=0.0,
                yolo_copy_paste=0.0,
                yolo_hsv_h=0.0,
                yolo_hsv_s=0.0,
                yolo_hsv_v=0.0,
            )
        )

    for config in configs:
        run_training(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run short training runs for Faster R-CNN, RetinaNet, and YOLO11."
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument(
        "--model",
        choices=["fasterrcnn", "retinanet", "yolo11", "all"],
        default="all",
    )
    args = parser.parse_args()

    run_smoke_training(num_epochs=args.epochs, model=args.model)
