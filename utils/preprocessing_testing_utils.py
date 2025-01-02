"""
Different testing functions used for the preprocessing pipeline.

These will not be used in the end product. They are just for debugging
and testing purposes.
"""

import sys
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

import matplotlib.pyplot as plt

def test_augmentation(augmented_images, grid_size=(2, 3), titles=None):
    num_images = len(augmented_images)
    rows, cols = grid_size
    
    # ensure the grid can fit all images
    num_plots = min(num_images, rows * cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 10))
    axes = axes.flatten()
    
    for i in range(num_plots):
        axes[i].imshow(augmented_images[i], cmap='gray')
        if titles and i < len(titles):
            axes[i].set_title(titles[i])
        else:
            axes[i].set_title(f"Image {i+1}")
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(num_plots, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def test_partitioning(client_data):
    for client_id, client_df in client_data.items():
        print(f"Distribution for {client_id}:")
        client_df.groupBy("Pneumonia").count().show()

def verify_unique_split(client_dfs):
    all_paths = set()
    duplicate_paths = set()

    for idx, client_df in enumerate(client_dfs):
        # Collect all paths for the current client
        paths = set(client_df.select("Augmented_Path").rdd.flatMap(lambda x: x).collect())

        # Check for duplicates with existing paths
        duplicates = all_paths.intersection(paths)
        if duplicates:
            duplicate_paths.update(duplicates)
        
        # Add current paths to the global set
        all_paths.update(paths)

    # Report results
    if duplicate_paths:
        print(f"Error: Duplicate paths found across clients: {duplicate_paths}")
    else:
        print("Success: No duplicate images across clients!")


