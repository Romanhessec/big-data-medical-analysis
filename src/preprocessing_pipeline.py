import pandas as pd
import os
import cv2
from pyspark.sql import SparkSession

def normalize_images(image_paths, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for path in image_paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = img / 255.0 # normalize to [0, 1]
        cv2.imwrite(os.path.join(output_folder, os.path.basename(path)), img)

def augment_image(image):
    pass

def partition_data():
    pass

# initialize spark session
spark = SparkSession.builder.appName("DataPartitioning").getOrCreate()

# load datasets labels
# WARNING: we do use test_labels.csv as val labels since test labels are 
# more numerous
val_df = pd.read_csv('../chexlocalize/CheXpert/test_labels.csv')
test_df = pd.read_csv("path/to/CheXpert/val_labels.csv")

# normalize data


# create distribution rules for each client
# this is a mock distribution - fix later
client_distribution = {
    'Client_1': {'Pneumonia': 0.6, 'Other': 0.4},
    'Client_2': {'Pneumonia': 0.3, 'Other': 0.7},
    'Client_3': {'Pneumonia': 0.5, 'Other': 0.5},
    'Client_4': {'Pneumonia': 0.4, 'Other': 0.6},
    'Client_5': {'Pneumonia': 0.2, 'Other': 0.8},
}

# partition the data

# verify distribution