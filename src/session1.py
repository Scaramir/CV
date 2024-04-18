# pip install dicom-parser

from dicom_parser import Image

img = Image('../data/series-00000/image-00000.dcm') # just the first one
# https://www.dicomlibrary.com/dicom/dicom-tags/

# get tag value by name
# ATTENTION: no white spaces allowed
image_type = img.header.raw["ImageType"].value

