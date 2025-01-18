# big-data-medical-analysis

## Dataset
The dataset used in this project is CheXpert. You can download it from [this link](https://stanfordaimi.azurewebsites.net/datasets/23c56a0d-15de-405b-87c8-99c30138950c).

After downloading, place the dataset in the root of this project directory.

## Run the pipelines
Pipeline should be run from the root directory: python3 src/preprocessing_pipeline.py && src/federated_learning_pipeline.py

## Requirements:
python-3.10 (or newer),
numpy-2.2.1,
pandas-2.2.3,
opencv-python-4.10.0.84,
pyspark-3.5.4,
matplotlib-3.10.0,
tensorflow-2.18.0,
