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
- why expect 91 classes? its the default of faterrcnn.
- set trainable_layers to maximum to train the whole model, or use a custom frozen backbone and train the rest of the model. we currently used 4 blocks and froze the first one. Allow all to be finetuned to see if it improves the model.
 - Classifier
   - ResNet50 (default, pretrtained on ImageNet1k)
   - Can we find a better pretraining / model and swap the backbone (and maybe its weights)?
     - pretrained xrv models. resnet50 weights as backbone weights and then finetune the model
 - RPN
   - default, but modified anchor sizes and ratios
   - use less anchors to speed up training

Loss-function / metrics: 
- why NaN loss?
  because update never got called

- How to weight/balance them?
  - calculate the ratio of the box_classes of our train_set after NMS and use this as weights for the loss function
  - adjust the class_predictions with the weights and overwrite the class_predictions in the loss_dict by writing a custom class
  - better option: sigmoid_focal_loss (Panopticnet uses it). Switch to it by changing jsut a few lines! 
- Sum(all losses)? yes
- mAP@IoU>=0.5 (PASCAL VOC 2010) as metric
  - Why mAP=0 when no box in an image predicted? But if no box predicted = target, we want to decrease the loss, right?
    - [X] Set 0011 box in target and prediction before calculating mAP to increase accuracy and decrease loss when no class is in the ground truth. 
    - 

