import pandas as pd
import os
import cv2
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, lit, rand
import numpy as np

def normalize_image(path, output_folder):
    """
    Normalize single image using Spark DataFrame
    Args:
        path: Image path
        output_folder: Folder to save normalized image
    Returns: 
        Normalized image.
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = img / 255.0
    new_path = os.path.join(output_folder, os.path.basename(path))
    cv2.imwrite(new_path, img)
    return new_path

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

    # apply parallel normalization
    normalize_udf = spark.udf.register("normalize_image", normalize_image)
    normalized_df = spark.df.withColumn("Normalized_path", normalize_udf(col("Path")))
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
spark = SparkSession.builder.appName("DataPartitioning").getOrCreate()

# load datasets labels
# WARNING: we do use test_labels.csv as val labels since test labels are 
# more numerous
val_df = pd.read_csv('../chexlocalize/CheXpert/test_labels.csv')
test_df = pd.read_csv("path/to/CheXpert/val_labels.csv")

# convert validation DataFrame to Spark DataFrame
spark_val_df = spark.createDataFrame(val_df)

# normalize data
normalized_val_df = normalize_images_spark(spark_val_df, "output/normalized_val")

# augment data
augmented_val_df = augment_images_spark(normalized_val_df, "output/augmented_val")

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
client_data = partition_data(spark, val_df, client_distribution)
for client, client_df in client_data.items():
    output_path = f"output/{client}/"
    os.makedirs(output_path, exist_ok=True)
    client_df.write.csv(os.path.join(output_path, "data.csv"), header=True)

# verify distribution
for client, client_df in client_data.items():
    print(f"Distribution for {client}:")
    client_df.groupBy("Pneumonia").count().show()