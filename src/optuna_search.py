import argparse
from pathlib import Path

import optuna

from amia import (
    AugmentationConfig,
    TrainingConfig,
    build_dataloaders_from_config,
    build_fasterrcnn_model,
    build_retinanet_model,
    train_and_evaluate,
    train_yolo11,
)


def suggest_augmentations(trial: optuna.Trial) -> AugmentationConfig:
    return AugmentationConfig(
        rotation_deg=trial.suggest_float("aug_rotation_deg", 0.0, 8.0),
        translate=trial.suggest_float("aug_translate", 0.0, 0.05),
        scale=trial.suggest_float("aug_scale", 0.0, 0.12),
        shear=trial.suggest_float("aug_shear", 0.0, 5.0),
        perspective=trial.suggest_float("aug_perspective", 0.0, 0.1),
        perspective_p=trial.suggest_float("aug_perspective_p", 0.0, 0.2),
        brightness=trial.suggest_float("aug_brightness", 0.0, 0.2),
        contrast=trial.suggest_float("aug_contrast", 0.0, 0.2),
        color_jitter_p=trial.suggest_float("aug_color_jitter_p", 0.4, 1.0),
        sharpness_factor=trial.suggest_float("aug_sharpness_factor", 1.0, 2.0),
        sharpness_p=trial.suggest_float("aug_sharpness_p", 0.0, 0.3),
        autocontrast_p=trial.suggest_float("aug_autocontrast_p", 0.0, 0.3),
        equalize_p=trial.suggest_float("aug_equalize_p", 0.0, 0.2),
        gaussian_blur_p=trial.suggest_float("aug_gaussian_blur_p", 0.0, 0.2),
        random_erasing_p=trial.suggest_float("aug_random_erasing_p", 0.0, 0.2),
        horizontal_flip_p=trial.suggest_float("aug_horizontal_flip_p", 0.0, 0.1),
    )


def build_trial_config(trial: optuna.Trial, model_type: str, seed: int) -> TrainingConfig:
    lr = trial.suggest_float("learning_rate", 1e-5, 3e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 3e-3, log=True)
    scheduler_gamma = trial.suggest_float("scheduler_gamma", 0.9, 0.99)
    num_epochs = trial.suggest_int("num_epochs", 1, 6)
    rpn_nms_thresh = trial.suggest_float("rpn_nms_thresh", 0.3, 0.7)
    box_score_thresh = trial.suggest_float("box_score_thresh", 0.05, 0.3)

    config = TrainingConfig(
        model_type=model_type,
        epochs=num_epochs,
        lr=lr,
        weight_decay=weight_decay,
        scheduler_gamma=scheduler_gamma,
        rpn_nms_thresh=rpn_nms_thresh,
        box_score_thresh=box_score_thresh,
        experiment_name=f"amia-optuna-{model_type}",
        run_name=f"trial-{trial.number}",
        log_with_mlflow=False,
        seed=seed,
    )
    config.augmentations = suggest_augmentations(trial)

    if model_type == "yolo11":
        config.yolo_mosaic = trial.suggest_float("yolo_mosaic", 0.0, 1.0)
        config.yolo_mixup = trial.suggest_float("yolo_mixup", 0.0, 0.2)
        config.yolo_copy_paste = trial.suggest_float("yolo_copy_paste", 0.0, 0.2)
        config.yolo_hsv_h = trial.suggest_float("yolo_hsv_h", 0.0, 0.04)
        config.yolo_hsv_s = trial.suggest_float("yolo_hsv_s", 0.0, 0.7)
        config.yolo_hsv_v = trial.suggest_float("yolo_hsv_v", 0.0, 0.7)
        config.yolo_fliplr = trial.suggest_float("yolo_fliplr", 0.0, 0.1)
        config.yolo_scale = trial.suggest_float("yolo_scale", 0.2, 0.9)
        config.yolo_translate = trial.suggest_float("yolo_translate", 0.0, 0.2)
        config.yolo_shear = trial.suggest_float("yolo_shear", 0.0, 0.1)
        config.yolo_perspective = trial.suggest_float("yolo_perspective", 0.0, 0.1)
        config.yolo_erasing = trial.suggest_float("yolo_erasing", 0.0, 0.2)

    return config


def objective(trial: optuna.Trial, model_type: str, seed: int) -> float:
    config = build_trial_config(trial, model_type, seed)
    dataloaders, num_classes, train_ids, test_ids = build_dataloaders_from_config(config)

    if model_type == "fasterrcnn":
        model = build_fasterrcnn_model(
            num_classes=num_classes,
            rpn_nms_thresh=config.rpn_nms_thresh,
            box_score_thresh=config.box_score_thresh,
        )
        _, _, map_history = train_and_evaluate(
            model,
            dataloaders["train"],
            dataloaders["test"],
            config=config,
            train_ids=train_ids,
            test_ids=test_ids,
        )
        return map_history[-1] if map_history else 0.0

    if model_type == "retinanet":
        model = build_retinanet_model(
            num_classes=num_classes,
            rpn_nms_thresh=config.rpn_nms_thresh,
            box_score_thresh=config.box_score_thresh,
        )
        _, _, map_history = train_and_evaluate(
            model,
            dataloaders["train"],
            dataloaders["test"],
            config=config,
            train_ids=train_ids,
            test_ids=test_ids,
        )
        return map_history[-1] if map_history else 0.0

    model, _, map_history, _ = train_yolo11(config, train_ids=train_ids, val_ids=test_ids)
    return map_history[-1] if map_history else 0.0


def run_study(model_type: str, trials: int, storage_path: Path, seed: int):
    storage = f"sqlite:///{storage_path.as_posix()}"
    study_name = f"amia-optuna-{model_type}"
    study = optuna.create_study(
        direction="maximize", study_name=study_name, storage=storage, load_if_exists=True
    )
    study.optimize(lambda trial: objective(trial, model_type, seed), n_trials=trials)
    return study


def print_best(model_type: str, storage_path: Path):
    storage = f"sqlite:///{storage_path.as_posix()}"
    study_name = f"amia-optuna-{model_type}"
    study = optuna.load_study(study_name=study_name, storage=storage)
    print(f"Best value ({model_type}): {study.best_value}")
    print(f"Best params ({model_type}): {study.best_params}")


def main():
    parser = argparse.ArgumentParser(
        description="Run Optuna hyperparameter search for AMIA models."
    )
    parser.add_argument(
        "--model",
        choices=["fasterrcnn", "retinanet", "yolo11", "all"],
        default="fasterrcnn",
    )
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument(
        "--study-dir", type=Path, default=Path("optuna_studies"), help="Storage folder"
    )
    parser.add_argument("--seed", type=int, default=123420)
    parser.add_argument("--print-best", action="store_true")
    args = parser.parse_args()

    args.study_dir.mkdir(parents=True, exist_ok=True)

    if args.model == "all":
        models = ["fasterrcnn", "retinanet", "yolo11"]
    else:
        models = [args.model]

    for model_type in models:
        storage_path = args.study_dir / f"optuna_{model_type}.db"
        if args.print_best:
            print_best(model_type, storage_path)
        else:
            run_study(model_type, args.trials, storage_path, args.seed)


if __name__ == "__main__":
    main()
