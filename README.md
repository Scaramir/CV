# AMIA Challenge

## Introduction
blabla

## Data
kaggle

## Methods
### Preprocessing
resizing
augmentations

### Model
FasterRCNN:
- why expect 91 classes?
- set trainable_layers to maximum to train the whole model, or use a custom frozen backbone and train the rest of the model.
 - Classifier
   - ResNet50 (default, pretrtained on ImageNet1k)
   - Can we find a better pretraining / model and swap the backbone (and maybe its weights)?

 - RPN
   - default, but modified anchors
   - 

Loss-function / metrics: 
- why NaN loss?
- Sum(all losses) 
- How to weight them?
- How to balance them?
  - How to handle the class imbalance?
- mAP@IoU>=0.5 (PASCAL VOC 2010) as metric
  - Why mAP=0 when no box in an image predicted? But if no box predicted = target, we want to decrease the loss, right?
    - [ ] Set 0011 box in target and prediction before calculating mAP to increase accuracy and dcrease loss when no class is in the ground truth. 





