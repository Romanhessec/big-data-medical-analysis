"""
Different testing functions used for the federated learning pipeline.

These will not be used in the end product. They are just for debugging
and testing purposes.
"""
import tensorflow as tf

def test_data_load(client_data):
    for client_id, dataset in client_data.items():
        print(f"{client_id}:")
        for batch_idx, data in enumerate(dataset):
            if isinstance(data, dict):  
                batch_size = tf.shape(list(data.values())[0])[0]  # assume all keys have the same batch size
            else:  # otherwise, handle data tuples (features, labels)
                batch_size = tf.shape(data[0])[0]
                
            print(f"Batch {batch_idx + 1}, Size: {batch_size.numpy()}")
