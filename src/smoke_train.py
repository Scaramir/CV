import argparse

from amia import TrainingConfig, run_training


def run_smoke_training(num_epochs=2):
    faster_config = TrainingConfig(
        model_type="fasterrcnn",
        epochs=num_epochs,
        batch_size=1,
        experiment_name="amia-smoke",
        run_name="fasterrcnn-smoke",
        train_limit=100,
        val_limit=100,
    )
    run_training(faster_config)

    retinanet_config = TrainingConfig(
        model_type="retinanet",
        epochs=num_epochs,
        batch_size=1,
        experiment_name="amia-smoke",
        run_name="retinanet-smoke",
        train_limit=100,
        val_limit=100,
    )
    run_training(retinanet_config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run short training runs for Faster R-CNN and RetinaNet."
    )
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()

    run_smoke_training(num_epochs=args.epochs)
