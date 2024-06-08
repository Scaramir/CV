import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.io import read_image
from torchvision.transforms import v2
import medmnist
from medmnist import ChestMNIST, DermaMNIST, INFO, Evaluator
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torch.optim import lr_scheduler
from tqdm.autonotebook import tqdm
from torcheval.metrics.functional import multiclass_confusion_matrix
from torchinfo import summary
import torchvision
from matplotlib import pyplot as plt
import numpy as np

import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms

import torch.optim as optim


import torchvision.transforms as transforms
import matplotlib.pyplot as plt


import random, warnings


# --------------- Helper functions ------------------
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


def get_mean_and_std(data_dir, print=False, leave_pbar=False):
    """
    Acquire the mean and std color values of all images (RGB-values) in the training set.
    inpupt: "data_dir" string
    output: mean and std Tensors
    """
    # Load the data set
    dataset = datasets.ImageFolder(
        os.path.join(data_dir, "train"), transform=transforms.ToTensor()
    )
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=1, num_workers=0
    )

    # Calculate the mean and std of the dataset
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _ in tqdm(
        dataloader,
        desc="Calculating mean and std of all RGB-values",
        leave=leave_pbar,
        colour="CYAN",
    ):
        non_black_pixels = data[(data != 0).any(dim=1)]
        channels_sum += torch.mean(non_black_pixels, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(non_black_pixels**2, dim=[0, 2, 3])
        num_batches += 1
    mean = channels_sum / num_batches
    # var[x] = E[x**2] - E[X]**2
    std = (channels_squared_sum / num_batches - mean**2) ** 0.5
    if print:
        print("Mean: ", mean, ", Std: ", std)
    return mean, std


# --------------- Data Loader ------------------
class XRayImageDataset(Dataset):
    def __init__(
        self,
        dataframe,
        img_dir,
        mean=None,
        std=None,
        target_transform=None,
    ):
        self.img_labels = dataframe
        self.img_dir = img_dir
        self.mean = mean
        self.std = std
        self.transform_norm = transforms.Compose(
            [
                transforms.Resize((224, 224), antialias=True),
                transforms.Normalize(self.mean, self.std),
            ]
        )
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_id = self.img_labels.iloc[idx]["image_id"]
        img_path = os.path.join(self.img_dir, img_id) + ".png"
        image = read_image(img_path).float()  # PyTorch function, no need to change
        label = self.img_labels.iloc[idx]["class_id"]  # class_id column
        image = self.transform_norm(image)
        return image, label


def load_and_augment_images(pic_folder_path, batch_size, use_normalize=True):
    if use_normalize:
        mean, std = get_mean_and_std(str(pic_folder_path))
        print("Mean: ", mean, ", Std: ", std)

    # Data augmentation and normalization for training
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.Resize((224, 224), antialias=True),
                transforms.RandomRotation(degrees=(-75, 75)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.0, hue=0.0
                ),
                transforms.ToTensor(),
                transforms.RandomPerspective(distortion_scale=0.3, p=0.3),
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.Resize((224, 224), antialias=True),
                transforms.ToTensor(),
            ]
        ),
    }
    if use_normalize:
        data_transforms["train"].transforms.append(
            transforms.Normalize(mean=mean, std=std, inplace=True)
        )
        data_transforms["test"].transforms.append(
            transforms.Normalize(mean=mean, std=std, inplace=True)
        )

    image_datasets = {
        x: datasets.ImageFolder(str(pic_folder_path / x), data_transforms[x])
        for x in ["train", "test"]
    }
    dataloaders = {
        x: torch.utils.data.DataLoader(
            image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=0
        )
        for x in ["train", "test"]
    }

    class_names = image_datasets["test"].classes
    num_classes = len(class_names)

    return dataloaders, class_names, num_classes
