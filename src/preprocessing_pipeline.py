import pandas as pd
import sys
import os
import cv2
import uuid
import numpy as np

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, concat, lit

# add project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.preprocessing_testing_utils import (
    test_normalization
)

def normalize_images_spark(spark_df, output_folder):
    """
    Normalize images using Spark DataFrame.
    Args:
        spark_df: Spark DataFrame containing image paths.
        output_folder: Folder to save normalized images.
    Returns:
        Normalized images DataFrame
    """
    os.makedirs(output_folder, exist_ok=True)

    def normalize_image_simple(path):
        try:
            if not os.path.exists(path):
                print(f"File does not exist: {path}")
                return None
            
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Failed to read image: {path}")
                return None
            
            img = cv2.equalizeHist(img)
            
            # generate unique identifier for filename
            unique_id = uuid.uuid4().hex
            base_name = os.path.basename(path)
            new_filename = f"{os.path.splitext(base_name)[0]}_{unique_id}.png"
            new_path = os.path.join(output_folder, new_filename)
            
            cv2.imwrite(new_path, img)
            return new_path
        except Exception as e:
            print(f"Error processing {path}: {e}")
            return None

    normalize_udf = udf(lambda path: normalize_image_simple(path))
    
    normalized_df = spark_df.withColumn("Normalized_path", normalize_udf(col("Path")))
    # force materialization
    normalized_df.select("Normalized_path").collect()

    return normalized_df
def augment_image(image):
    """
    Augment image with rotation, noise, contrast adjustment, etc.
    Args:
        image: Input image as a NumPy array.
    Returns:
        Augmented image as a NumPy array.
    """
    # random rotation
    angle = np.random.uniform(-15, 15)
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))

    # random contrast adjustment
    alpha = cv2.randu(0.8, 1.2)  # contrast
    beta = cv2.randu(-20, 20)    # brightness
    contrast_adjusted = cv2.convertScaleAbs(rotated, alpha=alpha, beta=beta)

    # add Gaussian noise
    noise = np.random.normal(0, 0.05, image.shape).astype(np.float32)
    noisy_image = np.clip(contrast_adjusted + noise, 0, 1)

    # random horizontal flip
    if np.random.rand() > 0.5:
        noisy_image = cv2.flip(noisy_image, 1)

    return noisy_image

def augment_images_spark(spark_df, output_folder):
    """
    Apply augmentation to images using Spark DataFrame.
    Args:
        spark_df: Spark DataFrame containing image paths.
        output_folder: Folder to save augmented images.
    """
    os.makedirs(output_folder, exist_ok=True)

    def augment_and_save(path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        augmented_img = augment_image(img)
        new_path = os.path.join(output_folder, os.path.basename(path))
        cv2.imwrite(new_path, augmented_img)
        return new_path

    # apply augmentation in parallel using Spark
    augment_udf = spark.udf.register("augment_and_save", augment_and_save)
    augmented_df = spark_df.withColumn("Augmented_Path", augment_udf(col("Path")))

    return augmented_df

def partition_data(spark, labels_df, client_distribution):
    """
    Partition data into non-IID subsets using Spark.
    Args:
        spark: Spark session.
        labels_df: Path to the CSV file or Pandas DataFrame with labels.
        client_distribution: Dictionary specifying label distribution for each client.
    Returns:
        A dictionary of Spark DataFrames for each client.
    """
    # convert panda df to spark df
    spark_df = spark.createDataFrame(labels_df)

    clients = {}

    for client, distribution in client_distribution.items():
        fractions = {label: distribution[label] for label in distribution}
        # should add 
        client_df = spark_df.sampleBy(col("Pneumonia"), fractions, seed = 42)
        clients[client] = client_df

    return clients

# initialize spark session
spark = SparkSession.builder \
    .appName("NormalizeImages") \
    .master("local[*]") \
    .getOrCreate()

# spark.sparkContext.setLogLevel("DEBUG") # delete later

# load datasets labels
# WARNING: we do use test_labels.csv as val labels since test labels are 
# more numerous
val_df = pd.read_csv("chexlocalize/CheXpert/test_labels.csv")
test_df = pd.read_csv("chexlocalize/CheXpert/val_labels.csv")

# convert validation DataFrame to Spark DataFrame
# the labels csv have relative paths - this is why we need base_dir
base_dir = "chexlocalize/CheXpert/"
spark_val_df_relative = spark.createDataFrame(val_df)
spark_val_df = spark_val_df_relative.withColumn(
    "Path",
    concat(lit(base_dir), col("Path"))
)

spark_val_df.show()

# set the number of partitions for Spark
spark_val_df = spark_val_df.repartition(16, col("Path"))

print(f"Number of partitions after repartitioning: {spark_val_df.rdd.getNumPartitions()}")
partition_sizes = spark_val_df.rdd.glom().map(len).collect()
print(f"Rows in each partition: {partition_sizes}")

# check for duplicates in partitions
duplicates = spark_val_df.groupBy("Path").count().filter("count > 1").count()
print(f"Number of duplicate rows: {duplicates}")
if duplicates > 0:
    spark_val_df = spark_val_df.dropDuplicates(["Path"])

# normalize data
output_dir = "output/normalized_val"
os.makedirs(output_dir, exist_ok=True)

normalized_val_df = normalize_images_spark(spark_val_df, output_dir)
normalized_val_df.show()

# Check number of processed rows
print(f"Total processed rows: {normalized_val_df.count()}")

# Verify output directory
output_files = os.listdir(output_dir)
print(f"Number of files in output directory: {len(output_files)}")

# Cross-check with Spark processed rows
processed_rows = normalized_val_df.count()
if len(output_files) != processed_rows:
    print("Mismatch: Processed rows do not match saved files.")
    print(f"Processed rows: {processed_rows}, Files in output: {len(output_files)}")

# test_normalization(normalized_val_df, output_dir)

# # augment data
# augmented_val_df = augment_images_spark(normalized_val_df, "output/augmented_val")

# # create distribution rules for each client
# # this is a mock distribution - fix later
# client_distribution = {
#     'Client_1': {'Pneumonia': 0.6, 'Other': 0.4},
#     'Client_2': {'Pneumonia': 0.3, 'Other': 0.7},
#     'Client_3': {'Pneumonia': 0.5, 'Other': 0.5},
#     'Client_4': {'Pneumonia': 0.4, 'Other': 0.6},
#     'Client_5': {'Pneumonia': 0.2, 'Other': 0.8},
# }

# # partition the data
# client_data = partition_data(spark, val_df, client_distribution)
# for client, client_df in client_data.items():
#     output_path = f"output/{client}/"
#     os.makedirs(output_path, exist_ok=True)
#     client_df.write.csv(os.path.join(output_path, "data.csv"), header=True)

# # verify distribution
# for client, client_df in client_data.items():
#     print(f"Distribution for {client}:")
#     client_df.groupBy("Pneumonia").count().show()