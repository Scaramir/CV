import argparse
import json
import shutil
from pathlib import Path

import optuna
import torch

from amia import (
    AugmentationConfig,
    TrainingConfig,
    build_dataloaders_from_config,
    build_fasterrcnn_model,
    build_retinanet_model,
    config_to_dict,
    normalize_config,
    run_training,
    train_and_evaluate,
    train_yolo11,
)


def suggest_augmentations(trial: optuna.Trial) -> AugmentationConfig:
    rrc_scale_min = trial.suggest_float("aug_rrc_scale_min", 0.8, 1.0)
    rrc_scale_max = trial.suggest_float("aug_rrc_scale_max", rrc_scale_min, 1.0)
    rrc_ratio_min = trial.suggest_float("aug_rrc_ratio_min", 0.85, 1.0)
    rrc_ratio_max = trial.suggest_float("aug_rrc_ratio_max", rrc_ratio_min, 1.15)
    gamma_min = trial.suggest_float("aug_gamma_min", 0.85, 1.0)
    gamma_max = trial.suggest_float("aug_gamma_max", gamma_min, 1.2)
    noise_std_min = trial.suggest_float("aug_noise_std_min", 0.002, 0.01)
    noise_std_max = trial.suggest_float("aug_noise_std_max", noise_std_min, 0.03)
    return AugmentationConfig(
        random_resized_crop_p=trial.suggest_float("aug_random_resized_crop_p", 0.0, 0.4),
        random_resized_crop_scale=(rrc_scale_min, rrc_scale_max),
        random_resized_crop_ratio=(rrc_ratio_min, rrc_ratio_max),
        rotation_deg=trial.suggest_float("aug_rotation_deg", 0.0, 8.0),
        translate=trial.suggest_float("aug_translate", 0.0, 0.05),
        scale=trial.suggest_float("aug_scale", 0.0, 0.12),
        shear=trial.suggest_float("aug_shear", 0.0, 5.0),
        perspective=trial.suggest_float("aug_perspective", 0.0, 0.1),
        perspective_p=trial.suggest_float("aug_perspective_p", 0.0, 0.2),
        brightness=trial.suggest_float("aug_brightness", 0.0, 0.2),
        contrast=trial.suggest_float("aug_contrast", 0.0, 0.2),
        color_jitter_p=trial.suggest_float("aug_color_jitter_p", 0.4, 1.0),
        gamma_p=trial.suggest_float("aug_gamma_p", 0.0, 0.2),
        gamma_range=(gamma_min, gamma_max),
        sharpness_factor=trial.suggest_float("aug_sharpness_factor", 1.0, 2.0),
        sharpness_p=trial.suggest_float("aug_sharpness_p", 0.0, 0.3),
        autocontrast_p=trial.suggest_float("aug_autocontrast_p", 0.0, 0.3),
        equalize_p=trial.suggest_float("aug_equalize_p", 0.0, 0.2),
        gaussian_blur_p=trial.suggest_float("aug_gaussian_blur_p", 0.0, 0.2),
        gaussian_noise_p=trial.suggest_float("aug_gaussian_noise_p", 0.0, 0.2),
        gaussian_noise_std_range=(noise_std_min, noise_std_max),
        random_erasing_p=trial.suggest_float("aug_random_erasing_p", 0.0, 0.2),
        horizontal_flip_p=trial.suggest_float("aug_horizontal_flip_p", 0.0, 0.1),
    )


def build_trial_config(
    trial: optuna.Trial,
    model_type: str,
    seed: int,
    train_limit: int | None = None,
    val_limit: int | None = None,
) -> TrainingConfig:
    lr = trial.suggest_float("learning_rate", 1e-5, 3e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 3e-3, log=True)
    scheduler_gamma = trial.suggest_float("scheduler_gamma", 0.9, 0.99)
    num_epochs = trial.suggest_int("num_epochs", 1, 6)
    rpn_nms_thresh = trial.suggest_float("rpn_nms_thresh", 0.3, 0.7)
    box_score_thresh = trial.suggest_float("box_score_thresh", 0.05, 0.3)
    optimizer_name = trial.suggest_categorical("optimizer", ["adamw", "adamax", "sgd"])
    scheduler_name = trial.suggest_categorical(
        "scheduler", ["cosine", "onecycle", "exponential"]
    )
    warmup_epochs = trial.suggest_int("warmup_epochs", 0, 2)
    use_ema = trial.suggest_categorical("use_ema", [False, True])
    ema_decay = trial.suggest_float("ema_decay", 0.995, 0.9999)

    config = TrainingConfig(
        model_type=model_type,
        epochs=num_epochs,
        lr=lr,
        weight_decay=weight_decay,
        scheduler_gamma=scheduler_gamma,
        optimizer_name=optimizer_name,
        scheduler_name=scheduler_name,
        warmup_epochs=warmup_epochs,
        use_ema=use_ema,
        ema_decay=ema_decay,
        rpn_nms_thresh=rpn_nms_thresh,
        box_score_thresh=box_score_thresh,
        experiment_name=f"amia-optuna-{model_type}",
        run_name=f"trial-{trial.number}",
        log_with_mlflow=False,
        seed=seed,
        train_limit=train_limit,
        val_limit=val_limit,
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


def save_best_trial(
    model_type: str,
    map_value: float,
    config: TrainingConfig,
    params: dict,
    model,
    best_state: dict | None,
    checkpoint_dir: Path | None,
):
    if best_state is None or checkpoint_dir is None:
        return
    current_best = best_state.get("value")
    if current_best is not None and map_value <= current_best:
        return
    best_state["value"] = map_value
    meta = {
        "best_value": map_value,
        "params": params,
        "config": config_to_dict(config),
    }
    (checkpoint_dir / f"best_{model_type}.json").write_text(
        json.dumps(meta, indent=2)
    )

    if model_type in ("fasterrcnn", "retinanet"):
        torch.save(model.state_dict(), checkpoint_dir / f"best_{model_type}.pt")
        return

    save_dir = getattr(getattr(model, "trainer", None), "save_dir", None)
    if save_dir:
        best_weights = Path(save_dir) / "weights" / "best.pt"
        if best_weights.exists():
            shutil.copy2(best_weights, checkpoint_dir / f"best_{model_type}.pt")


def build_pruner(
    patience: int = 5,
    n_startup_trials: int = 5,
    n_warmup_steps: int = 1,
    interval_steps: int = 1,
):
    median = optuna.pruners.MedianPruner(
        n_startup_trials=n_startup_trials,
        n_warmup_steps=n_warmup_steps,
        interval_steps=interval_steps,
    )
    return optuna.pruners.PatientPruner(median, patience=patience)


def make_pruning_reporter(trial: optuna.Trial, metric: str):
    metric_key = metric.lower()

    def reporter(epoch: int, train_loss: float, map_value: float):
        if metric_key == "train_loss":
            value = -train_loss
        else:
            value = map_value
        trial.report(value, step=epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return reporter


def objective(
    trial: optuna.Trial,
    model_type: str,
    seed: int,
    train_limit: int | None = None,
    val_limit: int | None = None,
    best_state: dict | None = None,
    checkpoint_dir: Path | None = None,
    prune_metric: str = "map",
) -> float:
    config = build_trial_config(trial, model_type, seed, train_limit, val_limit)
    dataloaders, num_classes, train_ids, test_ids = build_dataloaders_from_config(config)
    reporter = make_pruning_reporter(trial, prune_metric)

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
            reporter=reporter,
        )
        map_value = map_history[-1] if map_history else 0.0
        save_best_trial(
            model_type, map_value, config, trial.params, model, best_state, checkpoint_dir
        )
        return map_value

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
            reporter=reporter,
        )
        map_value = map_history[-1] if map_history else 0.0
        save_best_trial(
            model_type, map_value, config, trial.params, model, best_state, checkpoint_dir
        )
        return map_value

    model, _, map_history, _ = train_yolo11(config, train_ids=train_ids, val_ids=test_ids)
    map_value = map_history[-1] if map_history else 0.0
    save_best_trial(
        model_type, map_value, config, trial.params, model, best_state, checkpoint_dir
    )
    return map_value


def run_study(
    model_type: str,
    trials: int,
    storage_path: Path,
    seed: int,
    train_limit: int | None = None,
    val_limit: int | None = None,
    prune_metric: str = "map",
    pruner_patience: int = 5,
    pruner_startup_trials: int = 5,
    pruner_warmup_steps: int = 1,
    pruner_interval_steps: int = 1,
):
    storage = f"sqlite:///{storage_path.as_posix()}"
    study_name = f"amia-optuna-{model_type}"
    checkpoint_dir = storage_path.parent / "best_models"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_state = {"value": None}
    pruner = build_pruner(
        patience=pruner_patience,
        n_startup_trials=pruner_startup_trials,
        n_warmup_steps=pruner_warmup_steps,
        interval_steps=pruner_interval_steps,
    )
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        pruner=pruner,
    )
    study.optimize(
        lambda trial: objective(
            trial,
            model_type,
            seed,
            train_limit,
            val_limit,
            best_state,
            checkpoint_dir,
            prune_metric,
        ),
        n_trials=trials,
    )
    return study


def print_best(model_type: str, storage_path: Path):
    storage = f"sqlite:///{storage_path.as_posix()}"
    study_name = f"amia-optuna-{model_type}"
    study = optuna.load_study(study_name=study_name, storage=storage)
    print(f"Best value ({model_type}): {study.best_value}")
    print(f"Best params ({model_type}): {study.best_params}")


def train_best(model_type: str, study_dir: Path, epochs_override: int | None = None):
    best_path = study_dir / "best_models" / f"best_{model_type}.json"
    if not best_path.exists():
        raise FileNotFoundError(
            f"Best config not found for {model_type}. Run an Optuna study first."
        )
    payload = json.loads(best_path.read_text())
    config_data = payload.get("config", {})
    augmentations = AugmentationConfig(**config_data.pop("augmentations", {}))
    config = TrainingConfig(**config_data, augmentations=augmentations)
    config = normalize_config(config)
    if epochs_override is not None:
        config.epochs = epochs_override
    run_training(config)


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
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument("--print-best", action="store_true")
    parser.add_argument("--train-best", action="store_true")
    parser.add_argument("--train-best-epochs", type=int, default=50)
    parser.add_argument("--prune-metric", choices=["map", "train_loss"], default="map")
    parser.add_argument("--pruner-patience", type=int, default=5)
    parser.add_argument("--pruner-startup-trials", type=int, default=5)
    parser.add_argument("--pruner-warmup-steps", type=int, default=1)
    parser.add_argument("--pruner-interval-steps", type=int, default=1)
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
            continue
        if args.train_best:
            train_best(model_type, args.study_dir, args.train_best_epochs)
            continue
        run_study(
            model_type,
            args.trials,
            storage_path,
            args.seed,
            args.train_limit,
            args.val_limit,
            args.prune_metric,
            args.pruner_patience,
            args.pruner_startup_trials,
            args.pruner_warmup_steps,
            args.pruner_interval_steps,
        )


if __name__ == "__main__":
    main()
