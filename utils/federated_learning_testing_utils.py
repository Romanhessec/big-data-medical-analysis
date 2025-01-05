"""
Different testing functions used for the federated learning pipeline.

These will not be used in the end product. They are just for debugging
and testing purposes.
"""
import tensorflow as tf
from tensorflow.keras.applications import ResNet50

def test_data_load(client_data):
    for client_id, dataset in client_data.items():
        print(f"{client_id}:")
        for batch_idx, data in enumerate(dataset):
            if isinstance(data, dict):  
                batch_size = tf.shape(list(data.values())[0])[0]  # assume all keys have the same batch size
            else:  # otherwise, handle data tuples (features, labels)
                batch_size = tf.shape(data[0])[0]
                
            print(f"Batch {batch_idx + 1}, Size: {batch_size.numpy()}")

def test_one_batch(client_datasets, keras_model):
    for client_id, dataset in client_datasets.items():
        for batch in dataset.take(1):
            images, features = batch  # Images as input; features as disease labels
            print(f"Client: {client_id}, Images shape: {images.shape}, Features: {features.numpy()}")
            predictions = keras_model(images)  # Use the images to generate predictions
            print(f"Predictions: {predictions.numpy()}")
            break