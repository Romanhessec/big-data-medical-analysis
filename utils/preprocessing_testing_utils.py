"""
Different testing functions used for the preprocessing pipeline.

These will not be used in the end product. They are just for debugging
and testing purposes.
"""

import pandas as pd
import os
import cv2
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, lit, rand
import numpy as np
import matplotlib.pyplot as plt

def test_normalization(normalized_val_df, output_dir):
    first_image_path = normalized_val_df.select("Normalized_Path").first()["Normalized_Path"]

    img = cv2.imread(first_image_path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        print(f"Pixel range for {first_image_path}: Min={img.min()}, Max={img.max()}")
        plt.imshow(img, cmap='gray')
        plt.title("Normalized Image")
        plt.show()
    else:
        print(f"Image not found: {first_image_path}")