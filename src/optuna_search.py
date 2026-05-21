import argparse

import optuna

from amia import (
    batch_size,
    build_fasterrcnn_model,
    class_names,
    dict_path,
    inf_folder_path,
    load_and_augment_images,
    pic_folder_path,
    train_and_evaluate,
)

_cached_dataloaders = None
_cached_num_classes = None


def get_dataloaders():
    global _cached_dataloaders, _cached_num_classes
    if _cached_dataloaders is None or _cached_num_classes is None:
        _cached_dataloaders, _, _cached_num_classes = load_and_augment_images(
            pic_folder_path, inf_folder_path, dict_path, batch_size, class_names
        )
    return _cached_dataloaders, _cached_num_classes


def objective(trial):
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    num_epochs = trial.suggest_int("num_epochs", 1, 5)

    dataloaders, num_classes = get_dataloaders()
    model = build_fasterrcnn_model(num_classes=num_classes)
    _, _, map_history = train_and_evaluate(
        model,
        dataloaders["train"],
        dataloaders["test"],
        num_epochs=num_epochs,
        lr=lr,
        weight_decay=weight_decay,
        experiment_name="amia-optuna",
        run_name=f"trial-{trial.number}",
    )
    return map_history[-1] if map_history else 0.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Optuna hyperparameter search for Faster R-CNN."
    )
    parser.add_argument("--trials", type=int, default=100)
    args = parser.parse_args()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.trials)
