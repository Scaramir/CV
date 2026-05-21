import argparse

from amia import (
    batch_size,
    build_fasterrcnn_model,
    build_retinanet_model,
    class_names,
    dict_path,
    inf_folder_path,
    load_and_augment_images,
    pic_folder_path,
    train_and_evaluate,
)


def run_smoke_training(num_epochs=2):
    dataloaders, _, num_classes = load_and_augment_images(
        pic_folder_path, inf_folder_path, dict_path, batch_size, class_names
    )

    faster_model = build_fasterrcnn_model(num_classes=num_classes)
    train_and_evaluate(
        faster_model,
        dataloaders["train"],
        dataloaders["test"],
        num_epochs=num_epochs,
        experiment_name="amia-smoke",
        run_name="fasterrcnn-smoke",
    )

    retinanet_model = build_retinanet_model(num_classes=num_classes)
    train_and_evaluate(
        retinanet_model,
        dataloaders["train"],
        dataloaders["test"],
        num_epochs=num_epochs,
        experiment_name="amia-smoke",
        run_name="retinanet-smoke",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run short training runs for Faster R-CNN and RetinaNet."
    )
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()

    run_smoke_training(num_epochs=args.epochs)
