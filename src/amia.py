import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.io import read_image
from torchvision.transforms import v2
import torchvision.tv_tensors as tv
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
import json

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


def get_mean_and_std(data_dir, img_list, print=False, leave_pbar=False):
    """
    Acquire the mean and std color values of all images (RGB-values) in the training set.
    inpupt: "data_dir" string
    output: mean and std Tensors
    """
    # Load the data set matching img_list
    dataset = datasets.ImageFolder(
        [id for id in os.listdir(data_dir) if id in img_list],
        transform=transforms.ToTensor()
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
        dict,
        img_size,
        img_dir,
        mean=None,
        std=None,
        transform_norm=None,
    ):
        self.dict = dict
        self.keys = list(dict.keys())
        self.img_size = img_size
        self.img_dir = img_dir
        self.mean = mean
        self.std = std
        self.transform_norm = transform_norm

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        img_id = self.keys[idx]
        img_path = os.path.join(self.img_dir, img_id) + ".png"
        image = read_image(img_path).float()  # PyTorch function, no need to change
        image = self.transform_norm(image)
        # https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
        

        box_list = []
        label_list = []
        area_list = []
        iscrowd_list = []
        # get all boxes of all radiologists
        # structure of the dict
        # img_id: {
        #     "classes": [
        #         class_id: [
        #            rad_id: [[bbox],[bbox]]]

        for class_id in self.dict[img_id]["classes"].items():
            for rad in class_id.items():
                for box in rad.items():
                    box = box * self.img_size # is a float, maybe to int needed
                    box_list.append(box)
                    label_list.append(class_id)
                    area_list.append((box[2]-box[0])*(box[3]-box[1])) # x_max-x_min * y_max-y_min
                    iscrowd_list.append(0)

        target = {
            "boxes": tv.BoundingBoxes(box_list),
            "labels": label_list,
            "image_id": idx,
            "area": torch.tensor(area_list),
            "iscrowd": torch.tensor(iscrowd_list)
        }

        return image, target, img_id


def load_and_augment_images(pic_folder_path, batch_size, use_normalize=True):
    # split folders into 70% train and 30% test by ids
    set_seeds()
    train_percent = 0.7
    train_ids = random.sample(os.listdir(pic_folder_path), int(train_percent*len(os.listdir(pic_folder_path))))
    test_ids = [id for id in os.listdir(pic_folder_path) if id not in train_ids]
    
    # normalize on all train images
    if use_normalize:
        mean, std = get_mean_and_std(pic_folder_path, train_ids, print=False, leave_pbar=True)
        print("Mean: ", mean, ", Std: ", std)
    
    # remove file extension
    train_ids = [id.split(".")[0] for id in train_ids]
    test_ids = [id.split(".")[0] for id in test_ids]

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
    
    # load image_dict.json
    dict_path = 'pre-pro/image_dict.json'

    with open(dict_path) as f:
        train_dict = json.load(f)

    # train_dict where keys match train_ids
    train_dict = {k: train_dict[k] for k in train_ids}
    test_dict = {k: train_dict[k] for k in test_ids}

    

    # size for images
    img_size = 224
    train_dataset = XRayImageDataset(
        train_dict,
        img_size,
        pic_folder_path,
        mean,
        std,
        data_transforms["train"]
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )

    test_dataset = XRayImageDataset(
        test_dict,
        img_size,
        pic_folder_path,
        mean,
        std,
        data_transforms["test"]
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    dataloaders = {"train": train_dataloader, "test": test_dataloader}

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
    num_classes = class_names.items().__len__()

    return dataloaders, class_names, num_classes
