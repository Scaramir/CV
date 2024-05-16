# pip install dicom-parser
from pathlib import Path

from dicom_parser import Image

img = Image('../data/series-00000/image-00000.dcm') # just the first one
# https://www.dicomlibrary.com/dicom/dicom-tags/

# get tag value by name
# ATTENTION: no white spaces allowed
image_type = img.header.raw["ImageType"].value

# Exercise
# Imaging Technique? -> CT
img.header.raw["Modality"].value
# Body Part? -> abdomen
img.header.raw["StudyDescription"].value
# When image taken? -> 20061012
img.header.raw["AcquisitionDate"].value
# Patient Position in CT machine? -> FFS feet first supine
img.header.raw["PatientPosition"].value
# Code for Opposite Orientation? -> HFS head first supine
# XYZ Image size in millimeters?

# Retrieve Pixel Spacing
pixel_spacing = img.header.raw[(0x0028, 0x0030)].value # oh wow, tuples, ja? xD
pixel_spacing_x, pixel_spacing_y = map(float, pixel_spacing)
rows = img.header.raw["Rows"].value
columns = img.header.raw["Columns"].value
# number of dcm files in the /data folder?

# get number of slices in the folder
slices = len(list(Path("../data/series-00000").glob("*.dcm")))

# spacing between slices
spacing = img.header.raw[(0x0018, 0x0088)].value
x = rows * pixel_spacing_x
y = columns * pixel_spacing_y
z = slices * spacing

print(f"Image size: {x} x {y} x {z} mm")

# tag for patient name? do we have that information?
img.header.raw["PatientName"].value # Anonymized here

# didn't get to do this
# compression format for pixel data?
# bit depth?


# ---- kaggel section ----

# open and print image efficiently using PIL
from PIL import Image as PILImage
import matplotlib.pyplot as plt
import torch 
from torch.utils.data import Dataset

# dataset class for png images to load efficiently using torch
# __init__ takes all paths, mean and std
# __lean__ returns the lenght of the dataset 
# __getitem__ returns the normalized image and one-hot-encoded label
# Dataset object goes into torch.utils.data.DataLoader

class ImageDataset(Dataset):

    pass