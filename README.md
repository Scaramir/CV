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
micromamba create -f environment.yaml -n amia
micromamba activate amia
uv pip install -e .
```

## Experiment tracking (MLflow)
Launch the MLflow UI from the repo root:
```bash
mlflow ui
```
By default, runs are stored in `mlruns/` (gitignored) and metadata in `mlflow.db`.

## Training entry point
Run a training session from the repo root (uses 2026 data by default):
```bash
python src\amia.py --model fasterrcnn --epochs 2
```
Fine-tune YOLO11 (exports a YOLO dataset under `data\yolo11` on first run):
```bash
python src\amia.py --model yolo11 --epochs 50 --yolo-model yolo11s.pt --yolo-rebuild-dataset
```
To regenerate the image dictionary JSON from `train.csv` and `img_size.csv`:
```bash
python src\amia.py --rebuild-image-dict
```

## Smoke training runs
Run short 2-epoch trainings for Faster R-CNN and RetinaNet:
```bash
python src\smoke_train.py --epochs 2
```

## Optuna hyperparameter search
Kick off Optuna studies for all models:
```bash
python src\optuna_search.py --model all --trials 100 --study-dir optuna_studies
```
Print the best parameter combination found by Optuna for each model type:
```bash
python src\optuna_search.py --model fasterrcnn --print-best --study-dir optuna_studies
python src\optuna_search.py --model retinanet --print-best --study-dir optuna_studies
python src\optuna_search.py --model yolo11 --print-best --study-dir optuna_studies
```
