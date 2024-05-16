# !pip install medmnist
# !pip install torcheval


import medmnist
from medmnist import ChestMNIST, DermaMNIST, INFO, Evaluator

import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import v2
from torch.optim import lr_scheduler

from tqdm.autonotebook import tqdm
import torch.optim as optim
from torcheval.metrics.functional import multiclass_confusion_matrix
from torchinfo import summary

import torchvision

import torchvision.transforms as transforms
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.pyplot as plt


BATCH_SIZE = 32

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5), (0.5))]
)

trainset = torchvision.datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
)

testset = torchvision.datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
)

# functions to show an image


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
classes = np.arange(10).astype("str")
print(" ".join(f"{classes[labels[j]]:5s}" for j in range(BATCH_SIZE)))


lr = 0.001
out_size = 10
batch_size = 32


image_shape = {"width": 28, "hight": 28, "dims": 1}


class CNN(nn.Module):
    def __init__(
        self, image_shape={"width": 28, "hight": 28, "dims": 1}, out_size=out_size
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(image_shape["dims"], 16, 5, stride=1, padding="same")
        self.conv2 = nn.Conv2d(16, 32, 3, stride=1, padding="same")
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 14 * 14, 100)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(100, out_size)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


model = CNN(image_shape, out_size)
print(summary(model, input_size=(1, 1, 28, 28)))

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)


for epoch in range(20):  # loop over the dataset multiple times

    running_loss = 0.0
    for inputs, labels in tqdm(trainloader):
        # get the inputs; data is a list of [inputs, labels]

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
    print(f"[{epoch + 1}, loss: {running_loss / 1875:.3f}")
    running_loss = 0.0

print("Finished Training")
