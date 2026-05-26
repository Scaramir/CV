import argparse
import contextlib
import copy
import math
import os
import shutil
import time
from dataclasses import asdict, dataclass, field
from typing import Callable
from pathlib import Path
from tqdm.autonotebook import tqdm as tqdm
import torch
import pandas as pd
from torch.utils.data import Dataset
from torchvision import datasets, transforms, tv_tensors
from torchvision.io import read_image
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F2

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
import torch.utils.data as data
import torchvision.transforms as transforms

import torch.optim as optim

import torchvision.transforms as transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN, rpn
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchmetrics.detection.mean_ap import MeanAveragePrecision


import torchvision.transforms as transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN, rpn, RetinaNet
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import matplotlib.pyplot as plt

import random, warnings
import mlflow
import mlflow.pytorch
import torchxrayvision as xrv

# --------------- Paths & defaults ------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "amia-public-challenge-2026"
DEFAULT_TRAIN_DIR = DEFAULT_DATA_ROOT / "train" / "train"
DEFAULT_TEST_DIR = DEFAULT_DATA_ROOT / "test" / "test"
DEFAULT_TRAIN_CSV = DEFAULT_DATA_ROOT / "train.csv"
DEFAULT_IMG_SIZE_CSV = DEFAULT_DATA_ROOT / "img_size.csv"
DEFAULT_IMAGE_DICT_PATH = DEFAULT_DATA_ROOT / "image_dict.json"
DEFAULT_MLFLOW_DB = REPO_ROOT / "mlflow.db"
DEFAULT_ARTIFACT_ROOT = REPO_ROOT / "mlruns"

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


@dataclass
class AugmentationConfig:
    random_resized_crop_p: float = 0.0
    random_resized_crop_scale: tuple[float, float] = (0.9, 1.0)
    random_resized_crop_ratio: tuple[float, float] = (0.9, 1.1)
    rotation_deg: float = 6.0
    translate: float = 0.02
    scale: float = 0.05
    shear: float = 2.0
    perspective: float = 0.05
    perspective_p: float = 0.1
    brightness: float = 0.1
    contrast: float = 0.1
    color_jitter_p: float = 1.0
    gamma_p: float = 0.0
    gamma_range: tuple[float, float] = (0.9, 1.1)
    sharpness_factor: float = 1.5
    sharpness_p: float = 0.1
    autocontrast_p: float = 0.1
    equalize_p: float = 0.05
    gaussian_blur_p: float = 0.05
    gaussian_blur_kernel: int = 3
    gaussian_blur_sigma: tuple[float, float] = (0.1, 1.0)
    gaussian_noise_p: float = 0.0
    gaussian_noise_std_range: tuple[float, float] = (0.005, 0.02)
    random_erasing_p: float = 0.1
    random_erasing_scale: tuple[float, float] = (0.02, 0.08)
    random_erasing_ratio: tuple[float, float] = (0.3, 3.3)
    horizontal_flip_p: float = 0.0


@dataclass
class TrainingConfig:
    model_type: str = "fasterrcnn"
    epochs: int = 50
    batch_size: int = 24
    lr: float = 0.0005
    weight_decay: float = 0.0005
    scheduler_gamma: float = 0.95
    optimizer_name: str = "adamw"
    scheduler_name: str = "cosine"
    warmup_epochs: int = 1
    warmup_start_factor: float = 0.1
    onecycle_pct_start: float = 0.3
    onecycle_div_factor: float = 25.0
    onecycle_final_div_factor: float = 1e4
    image_size: int = 448
    label_merge_nms_iou_thresh: float = 0.5
    rpn_nms_thresh: float = 0.5
    box_nms_thresh: float = 0.5
    box_score_thresh: float = 0.1
    train_split: float = 0.8
    seed: int = 123420
    train_limit: int | None = None
    val_limit: int | None = None
    dataset_root: Path = DEFAULT_DATA_ROOT
    train_dir: Path = DEFAULT_TRAIN_DIR
    test_dir: Path = DEFAULT_TEST_DIR
    train_csv_path: Path = DEFAULT_TRAIN_CSV
    img_size_csv_path: Path = DEFAULT_IMG_SIZE_CSV
    image_dict_path: Path = DEFAULT_IMAGE_DICT_PATH
    rebuild_image_dict: bool = False
    experiment_name: str = "amia"
    run_name: str | None = None
    log_with_mlflow: bool = True
    augmentations: AugmentationConfig = field(default_factory=AugmentationConfig)
    use_amp: bool = True
    amp_dtype: str = "fp16"
    grad_clip_norm: float | None = 10.0
    grad_accum_steps: int = 1
    use_ema: bool = True
    ema_decay: float = 0.9998
    matmul_precision: str = "high"
    allow_tf32: bool = True
    deterministic: bool = True
    enable_cudnn_benchmark: bool = False
    use_channels_last: bool = False
    use_compile: bool = False
    auto_scale_batch_size: bool = True
    auto_scale_mode: str = "binsearch"
    auto_scale_steps_per_trial: int = 3
    auto_scale_max_trials: int = 10
    yolo_model: str = "yolo11s.pt"
    yolo_imgsz: int = 640
    yolo_dataset_root: Path = REPO_ROOT / "data" / "yolo11"
    yolo_rebuild_dataset: bool = False
    yolo_auto_batch: bool = True
    yolo_mosaic: float = 0.5
    yolo_mixup: float = 0.1
    yolo_copy_paste: float = 0.0
    yolo_hsv_h: float = 0.015
    yolo_hsv_s: float = 0.4
    yolo_hsv_v: float = 0.4
    yolo_fliplr: float = 0.0
    yolo_flipud: float = 0.0
    yolo_scale: float = 0.5
    yolo_translate: float = 0.1
    yolo_shear: float = 0.0
    yolo_perspective: float = 0.0
    yolo_erasing: float = 0.0
    yolo_conf_thresh: float = 0.01
    yolo_iou_thresh: float = 0.7
    yolo_warmup_epochs: float = 1.0


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
def set_seeds(seed=123420, deterministic=True, enable_cudnn_benchmark=False):
    random.seed(seed)
    np.random.seed(seed + 1)
    torch.random.manual_seed(seed + 2)
    device = get_device()
    if device == "cuda":
        torch.cuda.manual_seed(seed + 3)
        torch.cuda.manual_seed_all(seed + 4)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = enable_cudnn_benchmark and not deterministic
    print("Seeds set to {}.".format(seed))
    return


def configure_torch_backends(config: TrainingConfig):
    if config.matmul_precision:
        torch.set_float32_matmul_precision(config.matmul_precision)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = config.allow_tf32
        torch.backends.cudnn.allow_tf32 = config.allow_tf32


def resolve_amp_dtype(config: TrainingConfig) -> torch.dtype:
    amp_key = config.amp_dtype.lower()
    if amp_key in ("bf16", "bfloat16"):
        return torch.bfloat16
    if amp_key in ("fp16", "float16"):
        return torch.float16
    raise ValueError(f"Unsupported amp_dtype: {config.amp_dtype}")


def string_to_tensor(s):
    return torch.tensor([ord(c) for c in s], dtype=torch.int64)


def tensor_to_string(t):
    return "".join([chr(c) for c in t])



def normalize_config(config: TrainingConfig) -> TrainingConfig:
    config.dataset_root = Path(config.dataset_root)
    config.train_dir = Path(config.train_dir)
    config.test_dir = Path(config.test_dir)
    config.train_csv_path = Path(config.train_csv_path)
    config.img_size_csv_path = Path(config.img_size_csv_path)
    config.image_dict_path = Path(config.image_dict_path)
    config.yolo_dataset_root = Path(config.yolo_dataset_root)

    if config.dataset_root != DEFAULT_DATA_ROOT:
        if config.train_dir == DEFAULT_TRAIN_DIR:
            config.train_dir = config.dataset_root / "train" / "train"
        if config.test_dir == DEFAULT_TEST_DIR:
            config.test_dir = config.dataset_root / "test" / "test"
        if config.train_csv_path == DEFAULT_TRAIN_CSV:
            config.train_csv_path = config.dataset_root / "train.csv"
        if config.img_size_csv_path == DEFAULT_IMG_SIZE_CSV:
            config.img_size_csv_path = config.dataset_root / "img_size.csv"
        if config.image_dict_path == DEFAULT_IMAGE_DICT_PATH:
            config.image_dict_path = config.dataset_root / "image_dict.json"
    return config


def build_image_dict(train_csv_path: Path, img_size_csv_path: Path, output_path: Path):
    train_df = pd.read_csv(train_csv_path)
    train_df["class_id"] = train_df["class_id"].astype(int)
    img_size_df = pd.read_csv(img_size_csv_path)

    merged_df = pd.merge(train_df, img_size_df, on="image_id", how="inner")

    formatted_data: dict[str, dict] = {}
    for _, row in merged_df.iterrows():
        image_id = row["image_id"]
        og_height = row["dim0"]
        og_width = row["dim1"]
        class_id = str(row["class_id"])
        rad_id = str(row["rad_id"])

        if pd.isna(row["x_min"]):
            bounding_box = [0 / og_width, 0 / og_height, 1 / og_width, 1 / og_height]
        else:
            bounding_box = [
                row["x_min"] / og_width,
                row["y_min"] / og_height,
                row["x_max"] / og_width,
                row["y_max"] / og_height,
            ]

        if image_id not in formatted_data:
            formatted_data[image_id] = {"classes": {}}

        if class_id not in formatted_data[image_id]["classes"]:
            formatted_data[image_id]["classes"][class_id] = {}

        if rad_id not in formatted_data[image_id]["classes"][class_id]:
            formatted_data[image_id]["classes"][class_id][rad_id] = []

        formatted_data[image_id]["og_dims"] = [og_height, og_width]
        formatted_data[image_id]["classes"][class_id][rad_id].append(bounding_box)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as json_file:
        json.dump(formatted_data, json_file, indent=4)


def ensure_image_dict(config: TrainingConfig) -> Path:
    if config.rebuild_image_dict or not config.image_dict_path.exists():
        build_image_dict(config.train_csv_path, config.img_size_csv_path, config.image_dict_path)
    return config.image_dict_path


def config_to_dict(config: TrainingConfig) -> dict:
    data = asdict(config)
    for key in [
        "dataset_root",
        "train_dir",
        "test_dir",
        "train_csv_path",
        "img_size_csv_path",
        "image_dict_path",
        "yolo_dataset_root",
    ]:
        data[key] = str(data[key])
    return data


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


class InferenceImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_list = [
            f
            for f in os.listdir(img_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = read_image(img_path)
        if self.transform:
            image = self.transform(image)
        return image, Path(img_name).stem


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
    transform = v2.Compose(
        [v2.Grayscale(num_output_channels=1), v2.ToDtype(torch.float32, scale=True)]
    )
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

        canvas_size = (image.shape[-2], image.shape[-1])
        if len(box_list) > 0:
            boxes_tensor = tv_tensors.BoundingBoxes(
                torch.as_tensor(box_list, dtype=torch.float32),
                format="XYXY",
                canvas_size=canvas_size,
            )
        else:
            boxes_tensor = tv_tensors.BoundingBoxes(
                torch.zeros((0, 4), dtype=torch.float32),
                format="XYXY",
                canvas_size=canvas_size,
            )
        labels_tensor = torch.tensor(label_list, dtype=torch.int64)

        if self.transform_norm:
            image, boxes_tensor = self.transform_norm(image, boxes_tensor)

        boxes_tensor, labels_tensor, keep = sanitize_boxes_and_targets(
            boxes_tensor, labels_tensor, image
        )
        areas_tensor = compute_box_areas(boxes_tensor)
        iscrowd_tensor = torch.zeros_like(labels_tensor, dtype=torch.uint8)

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "area": areas_tensor,
            "iscrowd": iscrowd_tensor,
            "filename": string_to_tensor(img_id),
        }

        return image, target


def sanitize_boxes_and_targets(
    boxes_tensor: torch.Tensor,
    labels_tensor: torch.Tensor,
    image: torch.Tensor,
):
    if isinstance(boxes_tensor, tv_tensors.BoundingBoxes):
        canvas_size = boxes_tensor.canvas_size
    else:
        canvas_size = (image.shape[-2], image.shape[-1])

    height, width = image.shape[-2], image.shape[-1]
    if boxes_tensor.numel() == 0:
        empty_boxes = tv_tensors.BoundingBoxes(
            torch.zeros((0, 4), dtype=torch.float32),
            format="XYXY",
            canvas_size=canvas_size,
        )
        empty_labels = labels_tensor[:0]
        return empty_boxes, empty_labels, torch.zeros((0,), dtype=torch.bool)

    boxes = boxes_tensor.to(torch.float32).clone()
    boxes[:, 0].clamp_(0, width - 1)
    boxes[:, 2].clamp_(0, width - 1)
    boxes[:, 1].clamp_(0, height - 1)
    boxes[:, 3].clamp_(0, height - 1)

    keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    boxes = boxes[keep]
    labels_tensor = labels_tensor[keep]

    cleaned_boxes = tv_tensors.BoundingBoxes(
        boxes, format="XYXY", canvas_size=canvas_size
    )
    return cleaned_boxes, labels_tensor, keep


def compute_box_areas(boxes_tensor: torch.Tensor) -> torch.Tensor:
    if boxes_tensor.numel() == 0:
        return torch.zeros((0,), dtype=torch.float32)
    boxes = boxes_tensor.to(torch.float32)
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


class RandomGamma(v2.Transform):
    def __init__(
        self, gamma_range: tuple[float, float] = (0.9, 1.1), p: float = 0.1
    ):
        super().__init__()
        self.gamma_range = gamma_range
        self.p = p

    def transform(self, inpt, params):
        if isinstance(inpt, tv_tensors.BoundingBoxes):
            return inpt
        if torch.is_tensor(inpt):
            do = params.get("do") if params else None
            if do is None:
                do = float(torch.rand(1)) < self.p
            if not do:
                return inpt
            gamma = params.get("gamma") if params else None
            if gamma is None:
                gamma = float(
                    torch.empty(1).uniform_(self.gamma_range[0], self.gamma_range[1])
                )
            return F2.adjust_gamma(inpt, gamma)
        return inpt


class RandomGaussianNoise(v2.Transform):
    def __init__(
        self,
        std_range: tuple[float, float] = (0.005, 0.02),
        p: float = 0.1,
    ):
        super().__init__()
        self.std_range = std_range
        self.p = p

    def transform(self, inpt, params):
        if isinstance(inpt, tv_tensors.BoundingBoxes):
            return inpt
        if torch.is_tensor(inpt):
            do = params.get("do") if params else None
            if do is None:
                do = float(torch.rand(1)) < self.p
            if not do:
                return inpt
            std = params.get("std") if params else None
            if std is None:
                std = float(
                    torch.empty(1).uniform_(self.std_range[0], self.std_range[1])
                )
            noise = torch.randn_like(inpt) * std
            return torch.clamp(inpt + noise, 0.0, 1.0)
        return inpt


def build_xray_transforms(
    img_size: int, mean: float, std: float, augmentations: AugmentationConfig | None
):
    aug = augmentations or AugmentationConfig()

    if aug.random_resized_crop_p > 0:
        train_transforms: list = [
            v2.RandomApply(
                [
                    v2.RandomResizedCrop(
                        img_size,
                        scale=aug.random_resized_crop_scale,
                        ratio=aug.random_resized_crop_ratio,
                        antialias=True,
                    )
                ],
                p=aug.random_resized_crop_p,
            ),
            v2.Resize(img_size, antialias=True),
        ]
    else:
        train_transforms = [v2.Resize(img_size, antialias=True)]

    train_transforms += [
        v2.Grayscale(num_output_channels=1),
        v2.ToDtype(torch.float32, scale=True),
    ]

    if aug.horizontal_flip_p > 0:
        train_transforms.append(v2.RandomHorizontalFlip(p=aug.horizontal_flip_p))
    if aug.rotation_deg > 0:
        train_transforms.append(
            v2.RandomRotation(degrees=(-aug.rotation_deg, aug.rotation_deg))
        )
    if aug.translate > 0 or aug.scale > 0 or aug.shear > 0:
        scale_range = (max(0.0, 1.0 - aug.scale), 1.0 + aug.scale)
        train_transforms.append(
            v2.RandomAffine(
                degrees=0,
                translate=(aug.translate, aug.translate),
                scale=scale_range,
                shear=(-aug.shear, aug.shear),
            )
        )
    if aug.perspective > 0:
        train_transforms.append(
            v2.RandomPerspective(distortion_scale=aug.perspective, p=aug.perspective_p)
        )
    if aug.color_jitter_p > 0:
        train_transforms.append(
            v2.RandomApply(
                [
                    v2.ColorJitter(
                        brightness=aug.brightness, contrast=aug.contrast, saturation=0.0
                    )
                ],
                p=aug.color_jitter_p,
            )
        )
    if aug.gamma_p > 0:
        train_transforms.append(
            RandomGamma(gamma_range=aug.gamma_range, p=aug.gamma_p)
        )
    if aug.autocontrast_p > 0:
        train_transforms.append(v2.RandomAutocontrast(p=aug.autocontrast_p))
    if aug.equalize_p > 0:
        train_transforms.append(v2.RandomEqualize(p=aug.equalize_p))
    if aug.sharpness_p > 0:
        train_transforms.append(
            v2.RandomAdjustSharpness(
                sharpness_factor=aug.sharpness_factor, p=aug.sharpness_p
            )
        )
    if aug.gaussian_blur_p > 0:
        train_transforms.append(
            v2.RandomApply(
                [
                    v2.GaussianBlur(
                        kernel_size=aug.gaussian_blur_kernel,
                        sigma=aug.gaussian_blur_sigma,
                    )
                ],
                p=aug.gaussian_blur_p,
            )
        )
    if aug.gaussian_noise_p > 0:
        train_transforms.append(
            RandomGaussianNoise(
                std_range=aug.gaussian_noise_std_range, p=aug.gaussian_noise_p
            )
        )
    if aug.random_erasing_p > 0:
        train_transforms.append(
            v2.RandomErasing(
                p=aug.random_erasing_p,
                scale=aug.random_erasing_scale,
                ratio=aug.random_erasing_ratio,
                value="random",
            )
        )

    train_transforms.append(v2.Normalize(mean=[mean], std=[std]))

    test_transforms = v2.Compose(
        [
            v2.Resize(img_size, antialias=True),
            v2.Grayscale(num_output_channels=1),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[mean], std=[std]),
        ]
    )

    return {
        "train": v2.Compose(train_transforms),
        "test": test_transforms,
    }


def load_and_augment_images(
    pic_folder_path,
    inf_folder_path,
    dict_path,
    batch_size,
    class_names,
    img_size=448,
    use_normalize=False,
    train_split=0.8,
    nms_iou_thresh=0.5,
    seed=123420,
    train_limit=None,
    val_limit=None,
    augmentations: AugmentationConfig | None = None,
    deterministic: bool = True,
    enable_cudnn_benchmark: bool = False,
):
    # split folders into 70% train and 30% test by ids
    set_seeds(
        seed=seed,
        deterministic=deterministic,
        enable_cudnn_benchmark=enable_cudnn_benchmark,
    )
    train_percent = train_split
    # Use the images in the ONE folder and split them into train and test
    train_ids = random.sample(
        os.listdir(pic_folder_path),
        int(train_percent * len(os.listdir(pic_folder_path))),
    )
    test_ids = [id for id in os.listdir(pic_folder_path) if id not in train_ids]

    if train_limit is not None:
        train_ids = train_ids[:train_limit]
    if val_limit is not None:
        test_ids = test_ids[:val_limit]

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

    data_transforms = build_xray_transforms(img_size, mean, std, augmentations)

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
        train_dict,
        img_size,
        pic_folder_path,
        mean,
        std,
        data_transforms["train"],
        nms_iou_thresh=nms_iou_thresh,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    test_dataset = XRayImageDataset(
        test_dict,
        img_size,
        pic_folder_path,
        mean,
        std,
        data_transforms["test"],
        nms_iou_thresh=nms_iou_thresh,
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,  # no need for batches
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    # image folder
    inference_dataset = InferenceImageDataset(
        img_dir=inf_folder_path, transform=data_transforms["test"]
    )

    inference_dataloader = torch.utils.data.DataLoader(
        inference_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    print("Loaded the training dataset.")

    dataloaders = {
        "train": train_dataloader,
        "test": test_dataloader,
        "inference": inference_dataloader,
    }

    num_classes = class_names.items().__len__()

    return dataloaders, class_names, num_classes, train_ids, test_ids


# --------------- Model ------------------
# considering the bboxes of the k-means analysis, we chose the following anchors at each level. We rounded up to arbitrary values to end up a bit more in the upper right of each cluster.
# two cluster sizes were combined
def build_anchor_generator():
    return rpn.AnchorGenerator(
        sizes=((30,), (60,), (130,), (200,))
        * 5,  # times 5 'cause we want all sizes on all layers
        aspect_ratios=(
            (
                0.32,
                1.0,
                1.8,
            ),
        )
        * 5,  # times 5 'cause 5 feature maps'
    )


def build_backbone_with_fpn():
    # use pre-trained on chest x-rays
    backbone = xrv.models.ResNet(weights="resnet50-res512-all")
    backbone = torch.nn.Sequential(*list(backbone.model.children())[:-2])

    def fasterrcnn_reshape_transform(x):
        # Reshape the output of the FasterRCNN model to a format that can be used for visualization and evaluation purposes (EigenCam )
        target_size = x["pool"].size()[-2:]
        activations = []
        for _, value in x.items():
            activations.append(
                torch.nn.functional.interpolate(
                    torch.abs(value), target_size, mode="bilinear"
                )
            )
        activations = torch.cat(activations, axis=1)
        return activations

    # Define the layers to return feature maps from
    return_layers = {
        "4": "0",  # Corresponds to layer1
        "5": "1",  # Corresponds to layer2
        "6": "2",  # Corresponds to layer3
        "7": "3",  # Corresponds to layer4
        # TODO where is number '4'? roi_align has #5 feature maps
    }

    # Construct the BackboneWithFPN
    backbone_with_fpn = BackboneWithFPN(
        backbone,
        return_layers=return_layers,  # The layers we want to use
        in_channels_list=[
            256,
            512,
            1024,
            2048,
        ],  # Corresponding in_channels for these layers
        out_channels=256,  # Out channels for FPN layers
    )
    return backbone_with_fpn


def build_fasterrcnn_model(
    num_classes=15, rpn_nms_thresh=0.5, box_nms_thresh=0.5, box_score_thresh=0.1
):
    anchor_generator = build_anchor_generator()
    backbone_with_fpn = build_backbone_with_fpn()
    roi_align = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=["0", "1", "2", "3", "4"], output_size=7, sampling_ratio=2
    )
    model = torchvision.models.detection.FasterRCNN(
        backbone_with_fpn,
        num_classes=num_classes,
        # min_size=448, # produces NaN losses
        # max_size=448,
        # image_mean=[0.57062465], # TODO: to prevent ImageNet normalizing, we can change this to m=0, s=1 and rely on our own normalization :)
        # image_std=[0.24919559],
        image_mean=[0],
        image_std=[1],
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_align,
        # box_batch_size_per_image=128,
        # rpn_pre_nms_top_n_train=2000,
        # rpn_post_nms_top_n_test=1000,
        # rpn_post_nms_top_n_train=2000,
        # rpn_post_nms_top_n_test=1000,
        rpn_nms_thresh=rpn_nms_thresh,  # lower NMS -> fewer proposals
        box_nms_thresh=box_nms_thresh,
        box_score_thresh=box_score_thresh,  # increase to filter low-confidence detections
        box_detections_per_img=50,  # default 100 -> overkill?
    )
    return model


def build_retinanet_model(
    num_classes=15, box_nms_thresh=0.5, box_score_thresh=0.1
):
    anchor_generator = build_anchor_generator()
    backbone_with_fpn = build_backbone_with_fpn()
    model = RetinaNet(
        backbone_with_fpn,
        num_classes=num_classes,
        anchor_generator=anchor_generator,
        image_mean=[0],
        image_std=[1],
        score_thresh=box_score_thresh,
        nms_thresh=box_nms_thresh,
        detections_per_img=50,
    )
    return model


def build_yolo_class_mapping():
    mapping: dict[str, int] = {}
    names: list[str] = []
    for class_id in sorted(class_names.keys()):
        if class_id == 14:
            continue
        mapping[str(class_id)] = len(names)
        names.append(class_names[class_id])
    return mapping, names


def resolve_image_path(img_dir: Path, img_id: str) -> Path:
    for ext in (".png", ".jpg", ".jpeg"):
        candidate = img_dir / f"{img_id}{ext}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Image not found for id {img_id} in {img_dir}")


def link_or_copy_image(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def prepare_yolo_dataset(config: TrainingConfig, train_ids, val_ids) -> Path:
    yolo_root = config.yolo_dataset_root
    dataset_yaml = yolo_root / "dataset.yaml"
    if dataset_yaml.exists() and not config.yolo_rebuild_dataset:
        return dataset_yaml
    if config.yolo_rebuild_dataset and yolo_root.exists():
        shutil.rmtree(yolo_root)

    (yolo_root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (yolo_root / "images" / "val").mkdir(parents=True, exist_ok=True)
    (yolo_root / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (yolo_root / "labels" / "val").mkdir(parents=True, exist_ok=True)

    class_mapping, names = build_yolo_class_mapping()

    with open(config.image_dict_path) as f:
        og_dict = json.load(f)

    def write_split(split_name: str, ids):
        for img_id in ids:
            if img_id not in og_dict:
                raise KeyError(f"Image id {img_id} not found in image_dict.")
            src = resolve_image_path(config.train_dir, img_id)
            dst = yolo_root / "images" / split_name / src.name
            link_or_copy_image(src, dst)

            labels: list[str] = []
            for class_id, readers in og_dict[img_id]["classes"].items():
                if class_id == "14":
                    continue
                for _, boxes in readers.items():
                    for box in boxes:
                        if len(box) != 4 or box[2] <= box[0] or box[3] <= box[1]:
                            continue
                        x1, y1, x2, y2 = box
                        x_center = (x1 + x2) / 2
                        y_center = (y1 + y2) / 2
                        width = x2 - x1
                        height = y2 - y1
                        labels.append(
                            f"{class_mapping[class_id]} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                        )

            label_path = yolo_root / "labels" / split_name / f"{img_id}.txt"
            label_path.write_text("\n".join(labels))

    write_split("train", train_ids)
    write_split("val", val_ids)

    yaml_content = "\n".join(
        [
            f"path: {yolo_root.as_posix()}",
            "train: images/train",
            "val: images/val",
            f"nc: {len(names)}",
            f"names: {names}",
        ]
    )
    dataset_yaml.write_text(yaml_content)
    return dataset_yaml


def extract_yolo_map(model) -> float | None:
    trainer = getattr(model, "trainer", None)
    if trainer is None:
        return None
    metrics = getattr(trainer, "metrics", None)
    if metrics is None:
        return None
    if isinstance(metrics, dict):
        if "metrics/mAP50-95" in metrics:
            return float(metrics["metrics/mAP50-95"])
        if "metrics/mAP50(B)" in metrics:
            return float(metrics["metrics/mAP50(B)"])
    results_dict = getattr(metrics, "results_dict", None)
    if isinstance(results_dict, dict) and "metrics/mAP50-95" in results_dict:
        return float(results_dict["metrics/mAP50-95"])
    box_metrics = getattr(metrics, "box", None)
    if box_metrics is not None and hasattr(box_metrics, "map"):
        return float(box_metrics.map)
    return None


def train_yolo11(config: TrainingConfig, train_ids, val_ids, nested_run: bool = False):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "YOLO11 training requires ultralytics. Install it with `uv pip install ultralytics`."
        ) from exc

    dataset_yaml = prepare_yolo_dataset(config, train_ids, val_ids)
    model = YOLO(config.yolo_model)

    if config.log_with_mlflow:
        setup_mlflow(config)
        run_context = mlflow.start_run(run_name=config.run_name, nested=nested_run)
    else:
        run_context = contextlib.nullcontext()

    with run_context:
        if config.log_with_mlflow:
            mlflow.log_dict(config_to_dict(config), "config.json")
            mlflow.log_artifact(str(dataset_yaml), artifact_path="data")

        yolo_batch = -1 if config.yolo_auto_batch else config.batch_size
        results = model.train(
            data=str(dataset_yaml),
            imgsz=config.yolo_imgsz,
            epochs=config.epochs,
            batch=yolo_batch,
            lr0=config.lr,
            weight_decay=config.weight_decay,
            amp=config.use_amp,
            device=0 if torch.cuda.is_available() else "cpu",
            workers=num_workers,
            seed=config.seed,
            deterministic=config.deterministic,
            mosaic=config.yolo_mosaic,
            mixup=config.yolo_mixup,
            copy_paste=config.yolo_copy_paste,
            hsv_h=config.yolo_hsv_h,
            hsv_s=config.yolo_hsv_s,
            hsv_v=config.yolo_hsv_v,
            fliplr=config.yolo_fliplr,
            flipud=config.yolo_flipud,
            scale=config.yolo_scale,
            translate=config.yolo_translate,
            shear=config.yolo_shear,
            perspective=config.yolo_perspective,
            erasing=config.yolo_erasing,
            conf=config.yolo_conf_thresh,
            iou=config.yolo_iou_thresh,
            warmup_epochs=config.yolo_warmup_epochs,
            project=str(REPO_ROOT / "runs"),
            name=config.run_name or "yolo11",
            exist_ok=True,
        )

        if config.log_with_mlflow:
            save_dir = getattr(getattr(model, "trainer", None), "save_dir", None)
            if save_dir:
                mlflow.log_artifacts(str(save_dir), artifact_path="yolo")

    map_value = extract_yolo_map(model)
    map_history = [map_value] if map_value is not None else []
    return model, [], map_history, results


def plot_img_bbox(img, target, pred, title):
    # plot the image and bboxes
    # different colors for target and pred
    _, ax = plt.subplots(1, 1, figsize=(10, 10))
    img = img.cpu().permute(1, 2, 0)
    ax.imshow(img, cmap="gray")
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


# --------------- Training ------------------


class ModelEMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.9998, device=None):
        self.module = copy.deepcopy(model).eval()
        self.decay = decay
        if device is not None:
            self.module.to(device)
        for param in self.module.parameters():
            param.requires_grad_(False)

    def update(self, model: torch.nn.Module):
        with torch.no_grad():
            ema_state = self.module.state_dict()
            model_state = model.state_dict()
            for key, value in model_state.items():
                if key not in ema_state:
                    continue
                if not torch.is_floating_point(value):
                    ema_state[key] = value
                    continue
                ema_state[key].mul_(self.decay).add_(value, alpha=1.0 - self.decay)


def find_optimal_batch_size(
    config: TrainingConfig, num_classes: int, train_dataset: Dataset
) -> int:
    try:
        import lightning.pytorch as pl
        from lightning.pytorch.tuner import Tuner
    except ImportError as exc:
        raise ImportError(
            "Auto batch size finding requires lightning. Install it with `uv pip install lightning`."
        ) from exc

    class BatchSizeDataModule(pl.LightningDataModule):
        def __init__(self, dataset: Dataset, batch_size: int):
            super().__init__()
            self.dataset = dataset
            self.batch_size = batch_size

        def train_dataloader(self):
            return DataLoader(
                self.dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=num_workers,
                collate_fn=collate_fn,
            )

    class BatchSizeFinderModule(pl.LightningModule):
        def __init__(self, model: nn.Module, optimizer_name: str, lr: float, weight_decay: float):
            super().__init__()
            self.model = model
            self.optimizer_name = optimizer_name
            self.lr = lr
            self.weight_decay = weight_decay

        def training_step(self, batch, batch_idx):
            images, targets = batch
            loss_dict = self.model(images, targets)
            return sum(loss for loss in loss_dict.values())

        def configure_optimizers(self):
            if self.optimizer_name == "adamw":
                return torch.optim.AdamW(
                    self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
                )
            if self.optimizer_name == "adamax":
                return torch.optim.Adamax(
                    self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
                )
            if self.optimizer_name == "sgd":
                return torch.optim.SGD(
                    self.model.parameters(),
                    lr=self.lr,
                    momentum=0.9,
                    weight_decay=self.weight_decay,
                )
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")

    if config.model_type == "retinanet":
        base_model = build_retinanet_model(
            num_classes=num_classes,
            box_nms_thresh=config.box_nms_thresh,
            box_score_thresh=config.box_score_thresh,
        )
    else:
        base_model = build_fasterrcnn_model(
            num_classes=num_classes,
            rpn_nms_thresh=config.rpn_nms_thresh,
            box_nms_thresh=config.box_nms_thresh,
            box_score_thresh=config.box_score_thresh,
        )

    lightning_model = BatchSizeFinderModule(
        base_model, config.optimizer_name.lower(), config.lr, config.weight_decay
    )
    data_module = BatchSizeDataModule(train_dataset, config.batch_size)

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
    )
    tuner = Tuner(trainer)
    tuner.scale_batch_size(
        lightning_model,
        train_dataloaders=data_module,
        mode=config.auto_scale_mode,
        init_val=config.batch_size,
        steps_per_trial=config.auto_scale_steps_per_trial,
        max_trials=config.auto_scale_max_trials,
    )
    return int(data_module.batch_size)

# since we use fasterrcnn from torchvision, we can use the losses from the model
# we need to adjust the losses to include class weights
# overload the fastrcnn_loss function used by forward() to include class weights


def setup_mlflow(config: TrainingConfig):
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    else:
        mlflow.set_tracking_uri(f"sqlite:///{DEFAULT_MLFLOW_DB}")

    experiment = mlflow.get_experiment_by_name(config.experiment_name)
    if experiment is not None and experiment.lifecycle_stage != "active":
        config.experiment_name = f"{config.experiment_name}-revived"
        experiment = None
    if experiment is not None and "://" not in experiment.artifact_location:
        fallback_name = f"{config.experiment_name}-local"
        fallback = mlflow.get_experiment_by_name(fallback_name)
        if fallback is None:
            try:
                mlflow.create_experiment(
                    fallback_name, artifact_location=DEFAULT_ARTIFACT_ROOT.as_uri()
                )
            except MlflowException:
                pass
        config.experiment_name = fallback_name
        experiment = None

    if experiment is None:
        try:
            mlflow.create_experiment(
                config.experiment_name, artifact_location=DEFAULT_ARTIFACT_ROOT.as_uri()
            )
        except MlflowException:
            pass
    mlflow.set_experiment(config.experiment_name)


def train_and_evaluate(
    model,
    train_dataloader,
    val_dataloader,
    config: TrainingConfig,
    train_ids=None,
    test_ids=None,
    reporter: Callable[[int, float, float], None] | None = None,
    nested_run: bool = False,
):
    set_seeds(
        config.seed,
        deterministic=config.deterministic,
        enable_cudnn_benchmark=config.enable_cudnn_benchmark,
    )
    configure_torch_backends(config)
    torch.cuda.empty_cache()
    device = get_device()
    if config.use_amp and device == "cuda":
        torch.clear_autocast_cache()
    model.to(device)
    if config.use_channels_last and device == "cuda":
        model = model.to(memory_format=torch.channels_last)
    if config.use_compile and hasattr(torch, "compile"):
        model = torch.compile(model)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer_name = config.optimizer_name.lower()
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            params, lr=config.lr, weight_decay=config.weight_decay
        )
    elif optimizer_name == "adamax":
        optimizer = torch.optim.Adamax(
            params, lr=config.lr, weight_decay=config.weight_decay
        )
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            params, lr=config.lr, momentum=0.9, weight_decay=config.weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {config.optimizer_name}")

    scheduler_name = config.scheduler_name.lower()
    lr_scheduler = None
    step_scheduler_per_batch = False
    if scheduler_name == "cosine":
        if config.warmup_epochs > 0:
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=config.warmup_start_factor,
                total_iters=config.warmup_epochs,
            )
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, config.epochs - config.warmup_epochs),
            )
            lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup, cosine], milestones=[config.warmup_epochs]
            )
        else:
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, config.epochs)
            )
    elif scheduler_name == "onecycle":
        steps_per_epoch = max(
            1, math.ceil(len(train_dataloader) / max(1, config.grad_accum_steps))
        )
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.lr,
            epochs=config.epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=config.onecycle_pct_start,
            div_factor=config.onecycle_div_factor,
            final_div_factor=config.onecycle_final_div_factor,
        )
        step_scheduler_per_batch = True
    elif scheduler_name == "exponential":
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=config.scheduler_gamma
        )
    elif scheduler_name == "none":
        lr_scheduler = None
    else:
        raise ValueError(f"Unsupported scheduler: {config.scheduler_name}")

    # Initialize MeanAveragePrecision metric
    metric = MeanAveragePrecision(
        box_format="xyxy",
        iou_type="bbox",
        class_metrics=True,
        # iou_thresholds=[0.1],#[0.1, 0.4, 0.7],
    )

    train_losses = []
    map_history = []

    if config.log_with_mlflow:
        setup_mlflow(config)
        run_context = mlflow.start_run(run_name=config.run_name, nested=nested_run)
    else:
        run_context = contextlib.nullcontext()

    with run_context:
        if config.log_with_mlflow:
            mlflow.log_dict(config_to_dict(config), "config.json")
            mlflow.log_params(
                {
                    "model_type": config.model_type,
                    "num_epochs": config.epochs,
                    "learning_rate": config.lr,
                    "weight_decay": config.weight_decay,
                    "optimizer": config.optimizer_name,
                    "lr_scheduler": config.scheduler_name,
                    "scheduler_gamma": config.scheduler_gamma,
                    "warmup_epochs": config.warmup_epochs,
                    "warmup_start_factor": config.warmup_start_factor,
                    "onecycle_pct_start": config.onecycle_pct_start,
                    "onecycle_div_factor": config.onecycle_div_factor,
                    "onecycle_final_div_factor": config.onecycle_final_div_factor,
                    "model_name": model.__class__.__name__,
                    "train_size": len(train_dataloader.dataset),
                    "val_size": len(val_dataloader.dataset),
                    "train_batch_size": train_dataloader.batch_size,
                    "device": device,
                    "use_amp": config.use_amp,
                    "amp_dtype": config.amp_dtype,
                    "grad_clip_norm": config.grad_clip_norm,
                    "grad_accum_steps": config.grad_accum_steps,
                    "use_ema": config.use_ema,
                    "ema_decay": config.ema_decay,
                    "matmul_precision": config.matmul_precision,
                    "allow_tf32": config.allow_tf32,
                    "deterministic": config.deterministic,
                    "enable_cudnn_benchmark": config.enable_cudnn_benchmark,
                    "use_channels_last": config.use_channels_last,
                    "use_compile": config.use_compile,
                    "auto_scale_batch_size": config.auto_scale_batch_size,
                    "auto_scale_mode": config.auto_scale_mode,
                    "auto_scale_steps_per_trial": config.auto_scale_steps_per_trial,
                    "auto_scale_max_trials": config.auto_scale_max_trials,
                    "dataset_version": config.dataset_root.name,
                    "label_merge_nms_iou_thresh": config.label_merge_nms_iou_thresh,
                    "rpn_nms_thresh": config.rpn_nms_thresh,
                    "box_nms_thresh": config.box_nms_thresh,
                    "box_score_thresh": config.box_score_thresh,
                    "image_size": config.image_size,
                    "train_split": config.train_split,
                    "seed": config.seed,
                    "train_limit": config.train_limit,
                    "val_limit": config.val_limit,
                }
            )
            if config.image_dict_path.exists():
                mlflow.log_artifact(str(config.image_dict_path), artifact_path="data")
            if train_ids is not None and test_ids is not None:
                mlflow.log_dict(
                    {"train_ids": train_ids, "val_ids": test_ids}, "data/split.json"
                )

        print("Starting the training...")

        amp_dtype = resolve_amp_dtype(config)
        use_amp = config.use_amp and device == "cuda"
        use_scaler = use_amp and amp_dtype == torch.float16
        amp_device = "cuda" if device == "cuda" else "cpu"
        scaler = torch.amp.GradScaler(device=amp_device, enabled=use_scaler)
        ema = ModelEMA(model, decay=config.ema_decay, device=device) if config.use_ema else None
        for epoch in tqdm(range(config.epochs), desc="Epochs"):
            model.train()
            train_loss = 0
            loss_dict = {}
            loss_sums: dict[str, float] = {}
            epoch_start = time.perf_counter()
            processed_images = 0
            accum_steps = max(1, config.grad_accum_steps)
            accum_counter = 0
            optimizer.zero_grad(set_to_none=True)
            for images, targets in tqdm(
                train_dataloader, desc="Training", leave=True, colour="BLUE"
            ):
                processed_images += len(images)
                if config.use_channels_last and device == "cuda":
                    images = [
                        image.to(device, memory_format=torch.channels_last)
                        for image in images
                    ]
                else:
                    images = [image.to(device) for image in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                sanitized_images = []
                sanitized_targets = []
                for image, target in zip(images, targets):
                    boxes, labels, keep = sanitize_boxes_and_targets(
                        target["boxes"], target["labels"], image
                    )
                    if keep.numel() > 0 and not torch.all(keep):
                        warnings.warn(
                            f"Dropped {int((~keep).sum().item())} invalid boxes in training batch.",
                            UserWarning,
                        )
                    target["boxes"] = boxes
                    target["labels"] = labels
                    target["area"] = compute_box_areas(boxes).to(target["area"].device)
                    target["iscrowd"] = torch.zeros_like(
                        target["labels"], dtype=target["iscrowd"].dtype
                    )
                    sanitized_images.append(image)
                    sanitized_targets.append(target)
                images = sanitized_images
                targets = sanitized_targets

                # Apply mixed precision training
                with torch.amp.autocast(
                    device_type=amp_device, dtype=amp_dtype, enabled=use_amp
                ):
                    # when training the fasterrcnn model, the model returns a dict with losses.
                    # include class weights in the losses to balance the classes
                    loss_dict = model(images, targets)

                    losses = sum(loss for loss in loss_dict.values())
                    loss_value = losses.item()
                    loss_to_backprop = losses / accum_steps
                train_loss += loss_value
                for key, value in loss_dict.items():
                    loss_sums[key] = loss_sums.get(key, 0.0) + value.item()
                if use_scaler:
                    scaler.scale(loss_to_backprop).backward()
                else:
                    loss_to_backprop.backward()
                accum_counter += 1
                if accum_counter == accum_steps:
                    if config.grad_clip_norm:
                        if use_scaler:
                            scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(params, config.grad_clip_norm)
                    if use_scaler:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    if lr_scheduler and step_scheduler_per_batch:
                        lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    if ema:
                        ema.update(model)
                    accum_counter = 0
            if accum_counter > 0:
                if config.grad_clip_norm:
                    if use_scaler:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(params, config.grad_clip_norm)
                if use_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                if lr_scheduler and step_scheduler_per_batch:
                    lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                if ema:
                    ema.update(model)
            if lr_scheduler and not step_scheduler_per_batch:
                lr_scheduler.step()

            avg_train_loss = train_loss / max(1, len(train_dataloader))
            print(
                f"Epoch [{epoch+1}/{config.epochs}], Train Loss: {avg_train_loss:.4f}"
            )

            train_losses.append(avg_train_loss)
            epoch_time = max(1e-9, time.perf_counter() - epoch_start)
            samples_per_sec = processed_images / epoch_time
            if config.log_with_mlflow:
                mlflow.log_metric(
                    "train_samples_per_sec", samples_per_sec, step=epoch + 1
                )

            model.eval()
            eval_model = ema.module if ema else model
            with torch.no_grad():
                for images, targets in tqdm(
                    val_dataloader, desc="Validation", leave=True, colour="GREEN"
                ):
                    if config.use_channels_last and device == "cuda":
                        images = [
                            image.to(device, memory_format=torch.channels_last)
                            for image in images
                        ]
                    else:
                        images = [image.to(device) for image in images]
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                    sanitized_targets = []
                    for image, target in zip(images, targets):
                        boxes, labels, _ = sanitize_boxes_and_targets(
                            target["boxes"], target["labels"], image
                        )
                        target["boxes"] = boxes
                        target["labels"] = labels
                        target["area"] = compute_box_areas(boxes).to(
                            target["area"].device
                        )
                        target["iscrowd"] = torch.zeros_like(
                            target["labels"], dtype=target["iscrowd"].dtype
                        )
                        sanitized_targets.append(target)
                    targets = sanitized_targets
                    predictions = eval_model(images)

                    filtered_predictions = []
                    filtered_targets = []

                    # check if image gets predictions and plot
                    for img, target, pred in zip(images, targets, predictions):
                        if len(target["boxes"]) > 0:
                            # plot_img_bbox(
                            #     img,
                            #     target,
                            #     pred,
                            #     f"Image {tensor_to_string(target['filename'])}",
                            # )
                            filtered_predictions.append(pred)
                            filtered_targets.append(target)
                        elif (len(pred["boxes"]) == 0) and (len(target["boxes"]) == 0):
                            # append 1-pixel boxes
                            filtered_predictions.append(
                                {
                                    "boxes": torch.tensor([[0, 0, 1, 1]]).to(device),
                                    "scores": torch.tensor([1.0]).to(device),
                                    "labels": torch.tensor([0]).to(device),
                                }
                            )
                            filtered_targets.append(
                                {
                                    "boxes": torch.tensor([[0, 0, 1, 1]]).to(device),
                                    "scores": torch.tensor([1.0]).to(device),
                                    "labels": torch.tensor([0]).to(device),
                                }
                            )

                    # Calculate metrics
                    # if len(filtered_predictions) > 0:
                    metric.update(filtered_predictions, filtered_targets)
                    #     print(f"Filtered Outputs: {filtered_predictions}")
                    #     print(f"Filtered Targets: {filtered_targets}")

            # Calculate and print the mAP
            map_metric = metric.compute()
            map_value = (
                map_metric["map"].item()
                if isinstance(map_metric["map"], torch.Tensor)
                else float(map_metric["map"])
            )
            map_history.append(map_value)
            print(f"Epoch [{epoch+1}/{config.epochs}], Val mAP: {map_value:.4f}")
            print(map_metric)
            if reporter:
                reporter(epoch, avg_train_loss, map_value)

            if config.log_with_mlflow:
                mlflow.log_metric("train_loss", avg_train_loss, step=epoch + 1)
                mlflow.log_metric("val_map", map_value, step=epoch + 1)
                mlflow.log_metric(
                    "learning_rate", optimizer.param_groups[0]["lr"], step=epoch + 1
                )
                for key, total in loss_sums.items():
                    mlflow.log_metric(
                        f"train_{key}",
                        total / max(1, len(train_dataloader)),
                        step=epoch + 1,
                    )
                for key, value in map_metric.items():
                    if key == "map":
                        continue
                    if key.endswith("_per_class") and isinstance(value, torch.Tensor):
                        for idx, class_value in enumerate(value):
                            class_name = class_names.get(idx, f"class_{idx}")
                            metric_name = (
                                f"val_{key}_{class_name.lower().replace(' ', '_').replace('/', '_')}"
                            )
                            metric_value = (
                                class_value.item()
                                if class_value.is_floating_point()
                                else class_value.float().item()
                            )
                            mlflow.log_metric(metric_name, metric_value, step=epoch + 1)
                        continue
                    if isinstance(value, torch.Tensor):
                        value_tensor = (
                            value if value.is_floating_point() else value.float()
                        )
                        metric_value = (
                            value_tensor.item()
                            if value_tensor.numel() == 1
                            else value_tensor.mean().item()
                        )
                    else:
                        metric_value = float(value)
                    mlflow.log_metric(f"val_{key}", metric_value, step=epoch + 1)

            # Reset the metric for the next epoch
            metric.reset()
        if config.log_with_mlflow:
            mlflow.pytorch.log_model(model, "model")
    print("Finished Training!")
    return model, train_losses, map_history


def evaluate_and_create_csv(model, test_dataloader, device):
    model.eval()
    results = []

    with torch.no_grad():
        for images, image_ids in test_dataloader:
            images = [image.to(device) for image in images]
            outputs = model(images)

            for image_id, output in zip(image_ids, outputs):
                boxes = output["boxes"].cpu().numpy()
                labels = output["labels"].cpu().numpy()
                scores = output["scores"].cpu().numpy()

                row = [image_id]
                if len(boxes) > 0:
                    targets = []
                    for box, label, score in zip(boxes, labels, scores):
                        # Ensure the bounding box coordinates are integers
                        box = [int(b) for b in box]
                        # switch class back to AMIA format
                        targets.append(
                            f"{int(label)-1} {score:.2f} {int(box[0])} {int(box[1])} {int(box[2])} {int(box[3])}"
                        )
                    row.append(" ".join(targets))
                else:
                    # no boxes -> class 14 'No finding'
                    row.append("14 1 0 0 1 1")

                results.append(",".join(row))  # image_id, [class score box] [...]

    # Incorporate unix timestamp into the filename
    output_csv_path = f"submission_{int(time.time())}.csv"
    with open(output_csv_path, "w") as f:
        f.write("ID,TARGET\n")  # header row
        for result in results:
            f.write(f"{result}\n")

def build_dataloaders_from_config(config: TrainingConfig):
    image_dict_path = ensure_image_dict(config)
    dataloaders, _, num_classes, train_ids, test_ids = load_and_augment_images(
        str(config.train_dir),
        str(config.test_dir),
        str(image_dict_path),
        config.batch_size,
        class_names,
        img_size=config.image_size,
        use_normalize=False,
        train_split=config.train_split,
        nms_iou_thresh=config.label_merge_nms_iou_thresh,
        seed=config.seed,
        train_limit=config.train_limit,
        val_limit=config.val_limit,
        augmentations=config.augmentations,
        deterministic=config.deterministic,
        enable_cudnn_benchmark=config.enable_cudnn_benchmark,
    )
    return dataloaders, num_classes, train_ids, test_ids


def run_training(config: TrainingConfig):
    config = normalize_config(config)
    if config.auto_scale_batch_size and config.model_type != "yolo11":
        dataloaders, num_classes, train_ids, test_ids = build_dataloaders_from_config(
            config
        )
        config.batch_size = find_optimal_batch_size(
            config, num_classes, dataloaders["train"].dataset
        )
        dataloaders, num_classes, train_ids, test_ids = build_dataloaders_from_config(
            config
        )
    else:
        dataloaders, num_classes, train_ids, test_ids = build_dataloaders_from_config(
            config
        )

    if config.model_type == "yolo11":
        model, train_losses, map_history, _ = train_yolo11(
            config, train_ids=train_ids, val_ids=test_ids
        )
        return model, train_losses, map_history, dataloaders
    if config.model_type == "retinanet":
        model = build_retinanet_model(
            num_classes=num_classes,
            box_nms_thresh=config.box_nms_thresh,
            box_score_thresh=config.box_score_thresh,
        )
    else:
        model = build_fasterrcnn_model(
            num_classes=num_classes,
            rpn_nms_thresh=config.rpn_nms_thresh,
            box_nms_thresh=config.box_nms_thresh,
            box_score_thresh=config.box_score_thresh,
        )

    model, train_losses, map_history = train_and_evaluate(
        model,
        dataloaders["train"],
        dataloaders["test"],
        config=config,
        train_ids=train_ids,
        test_ids=test_ids,
    )
    return model, train_losses, map_history, dataloaders


def parse_args():
    parser = argparse.ArgumentParser(description="Train AMIA models with MLflow logging.")
    parser.add_argument(
        "--model", choices=["fasterrcnn", "retinanet", "yolo11"], default="fasterrcnn"
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--scheduler-gamma", type=float, default=0.95)
    parser.add_argument(
        "--optimizer", choices=["adamw", "adamax", "sgd"], default="adamw"
    )
    parser.add_argument(
        "--scheduler", choices=["cosine", "onecycle", "exponential", "none"], default="cosine"
    )
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--warmup-start-factor", type=float, default=0.1)
    parser.add_argument("--onecycle-pct-start", type=float, default=0.3)
    parser.add_argument("--onecycle-div-factor", type=float, default=25.0)
    parser.add_argument("--onecycle-final-div-factor", type=float, default=1e4)
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--amp-dtype", choices=["fp16", "bf16"], default="fp16")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--grad-clip-norm", type=float, default=None)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--ema-decay", type=float, default=0.9998)
    parser.add_argument(
        "--matmul-precision", choices=["highest", "high", "medium"], default="high"
    )
    parser.add_argument("--no-tf32", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--non-deterministic", dest="deterministic", action="store_false")
    parser.set_defaults(deterministic=True)
    parser.add_argument("--cudnn-benchmark", action="store_true")
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--no-auto-batch-size", dest="auto_batch_size", action="store_false")
    parser.set_defaults(auto_batch_size=True)
    parser.add_argument(
        "--auto-batch-size-mode", choices=["binsearch", "power"], default="binsearch"
    )
    parser.add_argument("--auto-batch-size-steps", type=int, default=3)
    parser.add_argument("--auto-batch-size-max-trials", type=int, default=10)
    parser.add_argument("--label-merge-nms-iou-thresh", type=float, default=0.5)
    parser.add_argument("--rpn-nms-thresh", type=float, default=0.5)
    parser.add_argument("--box-nms-thresh", type=float, default=0.5)
    parser.add_argument("--box-score-thresh", type=float, default=0.1)
    parser.add_argument("--train-split", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=123420)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--train-dir", type=Path, default=DEFAULT_TRAIN_DIR)
    parser.add_argument("--test-dir", type=Path, default=DEFAULT_TEST_DIR)
    parser.add_argument("--train-csv-path", type=Path, default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--img-size-csv-path", type=Path, default=DEFAULT_IMG_SIZE_CSV)
    parser.add_argument("--image-dict-path", type=Path, default=DEFAULT_IMAGE_DICT_PATH)
    parser.add_argument("--rebuild-image-dict", action="store_true")
    parser.add_argument("--yolo-model", default="yolo11s.pt")
    parser.add_argument("--yolo-imgsz", type=int, default=640)
    parser.add_argument("--yolo-dataset-root", type=Path, default=REPO_ROOT / "data" / "yolo11")
    parser.add_argument("--yolo-rebuild-dataset", action="store_true")
    parser.add_argument("--yolo-no-auto-batch", dest="yolo_auto_batch", action="store_false")
    parser.set_defaults(yolo_auto_batch=True)
    parser.add_argument("--yolo-conf-thresh", type=float, default=0.01)
    parser.add_argument("--yolo-iou-thresh", type=float, default=0.7)
    parser.add_argument("--yolo-warmup-epochs", type=float, default=1.0)
    parser.add_argument("--yolo-mosaic", type=float, default=0.5)
    parser.add_argument("--yolo-mixup", type=float, default=0.1)
    parser.add_argument("--yolo-copy-paste", type=float, default=0.0)
    parser.add_argument("--yolo-hsv-h", type=float, default=0.015)
    parser.add_argument("--yolo-hsv-s", type=float, default=0.4)
    parser.add_argument("--yolo-hsv-v", type=float, default=0.4)
    parser.add_argument("--yolo-fliplr", type=float, default=0.0)
    parser.add_argument("--yolo-flipud", type=float, default=0.0)
    parser.add_argument("--yolo-scale", type=float, default=0.5)
    parser.add_argument("--yolo-translate", type=float, default=0.1)
    parser.add_argument("--yolo-shear", type=float, default=0.0)
    parser.add_argument("--yolo-perspective", type=float, default=0.0)
    parser.add_argument("--yolo-erasing", type=float, default=0.0)
    parser.add_argument("--experiment-name", default="amia")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--no-mlflow", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = TrainingConfig(
        model_type=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        scheduler_gamma=args.scheduler_gamma,
        optimizer_name=args.optimizer,
        scheduler_name=args.scheduler,
        warmup_epochs=args.warmup_epochs,
        warmup_start_factor=args.warmup_start_factor,
        onecycle_pct_start=args.onecycle_pct_start,
        onecycle_div_factor=args.onecycle_div_factor,
        onecycle_final_div_factor=args.onecycle_final_div_factor,
        image_size=args.image_size,
        use_amp=not args.no_amp,
        amp_dtype=args.amp_dtype,
        grad_clip_norm=args.grad_clip_norm,
        grad_accum_steps=args.grad_accum_steps,
        use_ema=args.use_ema,
        ema_decay=args.ema_decay,
        matmul_precision=args.matmul_precision,
        allow_tf32=not args.no_tf32,
        deterministic=args.deterministic,
        enable_cudnn_benchmark=args.cudnn_benchmark,
        use_channels_last=args.channels_last,
        use_compile=args.compile,
        auto_scale_batch_size=args.auto_batch_size,
        auto_scale_mode=args.auto_batch_size_mode,
        auto_scale_steps_per_trial=args.auto_batch_size_steps,
        auto_scale_max_trials=args.auto_batch_size_max_trials,
        label_merge_nms_iou_thresh=args.label_merge_nms_iou_thresh,
        rpn_nms_thresh=args.rpn_nms_thresh,
        box_nms_thresh=args.box_nms_thresh,
        box_score_thresh=args.box_score_thresh,
        train_split=args.train_split,
        seed=args.seed,
        train_limit=args.train_limit,
        val_limit=args.val_limit,
        dataset_root=args.data_root,
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        train_csv_path=args.train_csv_path,
        img_size_csv_path=args.img_size_csv_path,
        image_dict_path=args.image_dict_path,
        rebuild_image_dict=args.rebuild_image_dict,
        yolo_model=args.yolo_model,
        yolo_imgsz=args.yolo_imgsz,
        yolo_dataset_root=args.yolo_dataset_root,
        yolo_rebuild_dataset=args.yolo_rebuild_dataset,
        yolo_auto_batch=args.yolo_auto_batch,
        yolo_conf_thresh=args.yolo_conf_thresh,
        yolo_iou_thresh=args.yolo_iou_thresh,
        yolo_warmup_epochs=args.yolo_warmup_epochs,
        yolo_mosaic=args.yolo_mosaic,
        yolo_mixup=args.yolo_mixup,
        yolo_copy_paste=args.yolo_copy_paste,
        yolo_hsv_h=args.yolo_hsv_h,
        yolo_hsv_s=args.yolo_hsv_s,
        yolo_hsv_v=args.yolo_hsv_v,
        yolo_fliplr=args.yolo_fliplr,
        yolo_flipud=args.yolo_flipud,
        yolo_scale=args.yolo_scale,
        yolo_translate=args.yolo_translate,
        yolo_shear=args.yolo_shear,
        yolo_perspective=args.yolo_perspective,
        yolo_erasing=args.yolo_erasing,
        experiment_name=args.experiment_name,
        run_name=args.run_name,
        log_with_mlflow=not args.no_mlflow,
    )
    run_training(config)


if __name__ == "__main__":
    main()
from mlflow.exceptions import MlflowException
