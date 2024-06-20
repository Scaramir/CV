import os
import pandas as pd
import numpy as np
import time
from tqdm.autonotebook import tqdm as tqdm
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms, tv_tensors
from torchvision.io import read_image
from torchvision.transforms import v2

# import medmnist
# from medmnist import ChestMNIST, DermaMNIST, INFO, Evaluator
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torch.optim import lr_scheduler, SGD
from tqdm.autonotebook import tqdm
from torcheval.metrics.functional import multiclass_confusion_matrix
from torchinfo import summary
import torchvision
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import json

import torch.nn.functional as F
from torchvision.transforms.v2 import functional as F2
import torch.utils.data as data
import torchvision.transforms as transforms

import torch.optim as optim


import torchvision.transforms as transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN, rpn
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import matplotlib.pyplot as plt

import random, warnings


# --------------- Constants ------------------

ROOT = "./../data/amia-public-challenge-2024/"
# call augment data function
pic_folder_path = ROOT + "train/train/"
inf_folder_path = ROOT + "test/test/"
dict_path = "./pre-pro/image_dict.json"
batch_size = 24

class_names = {
    0: "Aortic enlargement",
    1: "Atelectasis",
    2: "Calcification",
    3: "Cardiomegaly",
    4: "Consolidation",
    5: "ILD",
    6: "Infiltration",
    7: "Lung Opacity",
    8: "Nodule/Mass",
    9: "Other lesion",
    10: "Pleural effusion",
    11: "Pleural thickening",
    12: "Pneumothorax",
    13: "Pulmonary fibrosis",
    14: "No finding",
}

label_mapping = {
    "14": 0,  # 'No finding' mapped to 0 (background class)
    "0": 1,
    "1": 2,
    "2": 3,
    "3": 4,
    "4": 5,
    "5": 6,
    "6": 7,
    "7": 8,
    "8": 9,
    "9": 10,
    "10": 11,
    "11": 12,
    "12": 13,
    "13": 14,
}


# Detect OS and set num_workers accordingly
if os.name == "nt":  # Windows
    num_workers = 0
else:  # Linux and others
    num_workers = 2


# --------------- Helper functions ------------------
# Custom collate function to handle varying sizes of bounding boxes
def collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, targets


def get_device():
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
        warnings.warn("CUDA not available. Using CPU instead.", UserWarning)
    print("Device set to {}.".format(device))
    return device


# set seeds for reproducibility
def set_seeds(seed=123420):
    random.seed(seed)
    np.random.seed(seed + 1)
    torch.random.manual_seed(seed + 2)
    device = get_device()
    if device == "cuda":
        torch.cuda.manual_seed(seed + 3)
        torch.cuda.manual_seed_all(seed + 4)
        torch.backends.cudnn.deterministic = True
    print("Seeds set to {}.".format(seed))
    return


def string_to_tensor(s):
    return torch.tensor([ord(c) for c in s], dtype=torch.int64)


def tensor_to_string(t):
    return "".join([chr(c) for c in t])


class GrayscaleImageListDataset(Dataset):
    def __init__(self, img_dir, img_list, transform=None):
        self.img_dir = img_dir
        self.img_list = img_list
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_list[idx])
        image = read_image(img_path)
        if self.transform:
            image = self.transform(image)
        return image


def get_mean_and_std(
    img_dir, img_list, batch_size=32, print_values=False, leave_pbar=False
):
    """
    Compute the mean and std color values of all images (grayscale values) in the specified list.

    Parameters:
    - img_dir (str): Directory containing the images.
    - img_list (list): List of image filenames to include in the calculation.
    - batch_size (int): Batch size for processing images.
    - print_values (bool): Whether to print the mean and std values.
    - leave_pbar (bool): Whether to leave the progress bar after completion.

    Returns:
    - mean (torch.Tensor): Mean grayscale values.
    - std (torch.Tensor): Standard deviation of grayscale values.
    """
    device = get_device()
    transform = transforms.ToTensor()
    dataset = GrayscaleImageListDataset(img_dir, img_list, transform=transform)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )

    channels_sum = torch.zeros(1).to(device)
    channels_squared_sum = torch.zeros(1).to(device)
    num_pixels = 0

    for images in tqdm(
        dataloader,
        desc="Calculating mean and std of all grayscale values",
        leave=leave_pbar,
        colour="CYAN",
    ):
        images = images.to(device)
        non_black_pixels = images[images != 0].view(-1)
        num_pixels += non_black_pixels.shape[0]

        channels_sum += torch.sum(non_black_pixels)
        channels_squared_sum += torch.sum(non_black_pixels**2)

    mean = channels_sum / num_pixels
    std = (channels_squared_sum / num_pixels - mean**2) ** 0.5

    if print_values:
        print(
            "Mean: ", mean.cpu().detach().numpy(), ", Std: ", std.cpu().detach().numpy()
        )

    return mean, std


# --------------- Data Loader ------------------
class XRayImageDataset(Dataset):
    """
    load image and targets from dict
    structure of the dict
        img_id: {
            "classes": [
                class_id: [
                rad_id: [[bbox],[bbox]]]
    """

    def __init__(
        self,
        dict,
        img_size,
        img_dir,
        mean=None,
        std=None,
        transform_norm=None,
        nms=True,
        nms_iou_thresh=0.5,
    ):
        self.dict = dict
        self.keys = list(dict.keys())
        self.img_size = img_size
        self.img_dir = img_dir
        self.mean = mean
        self.std = std
        self.transform_norm = transform_norm
        self.nms = nms
        self.nms_iou_thresh = nms_iou_thresh

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        img_id = self.keys[idx]
        img_path = os.path.join(self.img_dir, img_id) + ".png"

        image = read_image(img_path)

        box_list = []
        label_list = []
        area_list = []
        iscrowd_list = []
        for class_id in self.dict[img_id]["classes"]:
            if class_id == "14":
                continue
            # collect all boxes for the current class
            box = []
            for rad in self.dict[img_id]["classes"][class_id].items():
                for box in rad[1]:
                    # Ensure the box has 4 coordinates
                    if len(box) == 4 and box[2] > box[0] and box[3] > box[1]:
                        box_list.append([coord * image.shape[-1] for coord in box])
                        label_list.append(label_mapping[class_id])
                        area_list.append((box[2] - box[0]) * (box[3] - box[1]))
                        iscrowd_list.append(0)
                    else:
                        print(f"Invalid Box found at {img_id} with {box}")
            if self.nms:
                # Non-maximum suppression
                boxes_to_keep = torchvision.ops.nms(
                    torch.tensor(box_list).float(),
                    torch.tensor([1.0] * len(box_list)),
                    iou_threshold=self.nms_iou_thresh,
                )
                # now keep only the values of the indices that are in boxes_to_keep
                box_list = [box_list[i] for i in boxes_to_keep]
                label_list = [label_list[i] for i in boxes_to_keep]
                area_list = [area_list[i] for i in boxes_to_keep]
                iscrowd_list = [iscrowd_list[i] for i in boxes_to_keep]

        if len(box_list) > 0:
            boxes_tensor = tv_tensors.BoundingBoxes(
                box_list, format="XYXY", canvas_size=(image.shape[-1], image.shape[-1])
            )
        else:
            empty_boxes = np.array([]).reshape(-1, 4)
            boxes_tensor = torch.as_tensor(empty_boxes, dtype=torch.int16)
        labels_tensor = torch.tensor(label_list, dtype=torch.int64)
        areas_tensor = torch.tensor(area_list, dtype=torch.int32)
        iscrowd_tensor = torch.tensor(iscrowd_list, dtype=torch.uint8)

        if self.transform_norm:
            image, boxes_tensor = self.transform_norm(image, boxes_tensor)

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "area": areas_tensor,
            "iscrowd": iscrowd_tensor,
            "filename": string_to_tensor(img_id),
        }

        return image, target


def load_and_augment_images(
    pic_folder_path,
    inf_folder_path,
    dict_path,
    batch_size,
    class_names,
    img_size=448,
    use_normalize=False,
):
    # split folders into 70% train and 30% test by ids
    set_seeds()
    train_percent = 0.8
    # Use the images in the ONE folder and split them into train and test
    train_ids = random.sample(
        os.listdir(pic_folder_path),
        int(train_percent * len(os.listdir(pic_folder_path))),
    )
    test_ids = [id for id in os.listdir(pic_folder_path) if id not in train_ids]

    # normalize on all train images or use precomputed
    if use_normalize:
        mean, std = get_mean_and_std(
            pic_folder_path, train_ids, print_values=True, leave_pbar=True
        )
        print("Mean: ", mean, ", Std: ", std)
    else:
        mean = 0.57062465
        std = 0.24919559

    # remove file extension
    train_ids = [id.split(".")[0] for id in train_ids]
    test_ids = [id.split(".")[0] for id in test_ids]
    # print first values and lengths
    print(f"Length of train_ids: {len(train_ids)}")
    print(f"Length of test_ids: {len(test_ids)}")

    # Data augmentation and normalization for training1
    data_transforms = {
        "train": v2.Compose(
            [
                v2.Resize(img_size, antialias=True),
                v2.RandomRotation(
                    degrees=(-6, 6)
                ),  # all images are upright and will always be. No rotation needed? COuld be interesting to try for generalizing
                v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.0, hue=0.0),
                v2.ToDtype(torch.float32, scale=False),
                v2.RandomPerspective(distortion_scale=0.1, p=0.1),
                v2.RandomEqualize(p=0.4),
                v2.Normalize(mean=[mean], std=[std], inplace=True),
            ]
        ),
        "test": v2.Compose(
            [
                v2.Resize(img_size, antialias=True),
                v2.ToDtype(torch.float32, scale=False),
                # Equalize all images to have a more uniform distribution of pixel intensities, regardless of they are generally dark or light
                v2.RandomEqualize(p=1.0),
                v2.Normalize(mean=[mean], std=[std], inplace=True),
            ]
        ),
    }

    # load image_dict.json
    with open(dict_path) as f:
        og_dict = json.load(f)

    # train_dict where keys match train_ids
    train_dict = {
        k: og_dict[k] for k in train_ids
    }  # if "14" not in og_dict[k]["classes"]}
    # print("Remaining train dict length: ", len(train_dict))
    test_dict = {
        k: og_dict[k] for k in test_ids
    }  # if "14" not in og_dict[k]["classes"]}
    # print("Remaining test dict length: ", len(test_dict))

    # size for images
    img_size = img_size
    train_dataset = XRayImageDataset(
        train_dict, img_size, pic_folder_path, mean, std, data_transforms["train"]
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    test_dataset = XRayImageDataset(
        test_dict, img_size, pic_folder_path, mean, std, data_transforms["test"]
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,  # no need for batches
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    # image folder
    inference_dataset = datasets.ImageFolder(
        root=inf_folder_path, transform=data_transforms["test"]
    )

    inference_dataloader = torch.utils.data.DataLoader(
        inference_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    print("Loaded the training dataset.")

    dataloaders = {"train": train_dataloader, "test": test_dataloader, "inference": inference_dataloader}

    num_classes = class_names.items().__len__()

    return dataloaders, class_names, num_classes


# --------------- Model ------------------
anchor_generator = rpn.AnchorGenerator(
    sizes=((32,), (48,), (64,), (96,), (128,)),
    aspect_ratios=((0.5, 1.0, 2.0),) * 5
)


model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")
model.rpn.anchor_generator = anchor_generator
# Get the number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features

# Replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(
    in_features, 15
)  # 14 classes + background
# NOTE: do we have to shift all classes so class 0 is 'No finding'? did this in the label_mapping
# TODO: Anchor boxes, how to set and adjust them?


def plot_img_bbox(img, target, pred, title):
    # plot the image and bboxes
    # different colors for target and pred
    _, ax = plt.subplots(1, 1, figsize=(10, 10))
    print(img.max())
    if img.max() > 1:
        img = img / 255.0
    img = img.cpu().permute(1, 2, 0)
    ax.imshow(img)
    for box in target["boxes"]:
        box = box.cpu().numpy()
        rect = plt.Rectangle(
            (box[0], box[1]),
            box[2] - box[0],
            box[3] - box[1],
            linewidth=2,
            edgecolor="g",
            facecolor="none",
        )
        ax.add_patch(rect)
    for box in pred["boxes"]:
        box = box.cpu().numpy()
        rect = plt.Rectangle(
            (box[0], box[1]),
            box[2] - box[0],
            box[3] - box[1],
            linewidth=2,
            edgecolor="r",
            facecolor="none",
        )
        ax.add_patch(rect)
    ax.set_title(title)
    # plt.savefig(f"{title}.png")


def train_and_evaluate(model, train_dataloader, val_dataloader, num_epochs=10, lr=0.05):
    device = get_device()
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)
    # optimizer = torch.optim.Adamax(params, lr=lr, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    exp_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95) # let's try this one as well

    # Initialize MeanAveragePrecision metric
    metric = MeanAveragePrecision(
        box_format="xyxy",
        iou_type="bbox",
        # iou_thresholds=[0.1],#[0.1, 0.4, 0.7],
    )

    print("Starting the training...")

    scaler = torch.cuda.amp.GradScaler()
    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        model.train()
        train_loss = 0
        inv_boxes = 0
        for images, targets in tqdm(
            train_dataloader, desc="Training", leave=True, colour="BLUE"
        ):
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            # Apply mixed precision training
            with torch.cuda.amp.autocast():
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
            train_loss += losses.item()

            optimizer.zero_grad()
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        lr_scheduler.step()  # TODO: adjust scheduler

        print(f"Invalid boxes: {inv_boxes}")

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}")

        model.eval()
        with torch.no_grad():
            for images, targets in tqdm(
                val_dataloader, desc="Validation", leave=True, colour="GREEN"
            ):
                images = [image.to(device) for image in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                predictions = model(images)

                filtered_predictions = []
                filtered_targets = []

                # check if image gets predictions and plot
                for img, target, pred in zip(images, targets, predictions):
                    if len(pred["boxes"]) > 0:
                        plot_img_bbox(
                            img,
                            target,
                            pred,
                            f"Image {tensor_to_string(target['filename'])}",
                        )
                        filtered_predictions.append(pred)
                        filtered_targets.append(target)

                # Calculate metrics
                if len(filtered_predictions) > 0:
                    metric.update(filtered_predictions, filtered_targets)
                    print(f"Filtered Outputs: {filtered_predictions}")
                    print(f"Filtered Targets: {filtered_targets}")

        # Calculate and print the mAP
        map_metric = metric.compute()
        print(f"Epoch [{epoch+1}/{num_epochs}], Val mAP: {map_metric['map']:.4f}")
        print(map_metric)

        # Reset the metric for the next epoch
        metric.reset()
    print("Finished Training!")


# --------------- Main ------------------

dataloaders, class_names, num_classes = load_and_augment_images(
    pic_folder_path, inf_folder_path, dict_path, batch_size, class_names
)
train_and_evaluate(model, dataloaders["train"], dataloaders["test"], 2)


def evaluate_and_create_csv(model, test_dataloader, device):
    model.eval()
    results = []
    
    with torch.no_grad():
        for images, image_ids in test_dataloader:
            images = [image.to(device) for image in images]
            outputs = model(images)
            
            for image_id, output in zip(image_ids, outputs):
                boxes = output['boxes'].cpu().numpy()
                labels = output['labels'].cpu().numpy()
                scores = output['scores'].cpu().numpy()
                
                row = [image_id]
                if len(boxes) > 0:
                    targets = []
                    for box, label, score in zip(boxes, labels, scores):
                        # Ensure the bounding box coordinates are integers
                        box = [int(b) for b in box]
                        # switch class back to AMIA format
                        targets.append(f"{int(label)-1} {score:.2f} {int(box[0])} {int(box[1])} {int(box[2])} {int(box[3])}")
                    row.append(" ".join(targets))
                else:
                    # no boxes -> class 14 'No finding'
                    row.append("14 1 0 0 1 1")
                
                results.append(",".join(row)) # image_id, [class score box] [...]
    
    # Incorporate unix timestamp into the filename
    output_csv_path = f'submission_{int(time.time())}.csv'
    with open(output_csv_path, 'w') as f:
        f.write("ID,TARGET\n") # header row
        for result in results:
            f.write(f"{result}\n")

evaluate_and_create_csv(model, dataloaders["inference"], get_device())

