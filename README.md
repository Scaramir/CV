# AMIA Kaggle Challenge

## Introduction
This repository contains our code for the AMIA Kaggle Challenge as part of the "Computer vision for biomedical images" seminar from Sören Lukassen @FU Berlin.  
We implemented some preprocessing steps to analyse the images and their labels, before doing a cluster analysis of the bounding boxes to obtain adequate anchor boxes for training different object detection models. 
The goal of this project was predicting up to 14 different findings in chest X-ray images.  
We trained FasterRCNN and RetinaNet with different hyperparameters to find the best model for the given dataset.  


A full report on how we tackled this project can be found [here](reports/CV_Lukassen_AMIA24_MO.pdf) (Max' report).  
Alternatively, an HTML version of the project report can be found [here](https://www.notion.so/floherzler/AMIA-Kaggle-Challenge-2024-51cf2f6466b3486eb70e3928f64acf56?pvs=4#7d12f74f9f1446a2b4bb931e576fa4c3) (Flo's report).


## Repository structure
- `data/`: contains the dataset and the preprocessed data
- `amia/`: contains figures and plots
- `src/`: contains the code for dataset exploration, preprocessing, training
- `runs/`: contains the trained models and their logs
- `reports/`: contains the project report in PDF format

## Usage
Use this for installation and execution of the code.
```bash 
# clone the repository first
# navigate to the repository
conda env create -f environment.yaml -n amia
conda activate amia
uv pip install -e .
```

## Experiment tracking (MLflow)
Start the MLflow UI from the repo root:
```bash
mlflow ui
```
Runs are stored in `mlruns/` (gitignored).

## Smoke training runs
Run short 2-epoch trainings for Faster R-CNN and RetinaNet:
```bash
python src\smoke_train.py --epochs 2
```

## Optuna hyperparameter search
Kick off an Optuna study for Faster R-CNN:
```bash
python src\optuna_search.py --trials 100
```
