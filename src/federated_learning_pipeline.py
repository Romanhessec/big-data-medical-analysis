import tensorflow as tf
import tensorflow_federated as tff
from tensorflow.keras.applications import MobileNetV2
from tensorflow_federated.python.learning.models import keras_utils
from tensorflow_federated.python.learning.optimizers import build_sgdm
import pandas as pd
import sys 
import os

# add project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.federated_learning_testing_utils import (
    test_data_load,
    test_one_batch
)

def load_evaluation_data(evaluation_data_path):
    """
    Load and preprocess the evaluation dataset.
    Args:
        evaluation_data_path: Path to the evaluation dataset CSV file.
    Returns:
        A TensorFlow dataset.
    """
    df = pd.read_csv(evaluation_data_path)

    # extract features and labels
    image_paths = df['Path'].values
    labels = df[['Cardiomegaly', 'Pneumonia', 'Lung Opacity', 'Edema', 'Consolidation']].values
    df['Path'] = df['Path'].str.replace('CheXpert-v1.0/valid', 'chexlocalize/CheXpert/val', regex=False)

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))

    # resize the images to adjust to model
    dataset = dataset.map(lambda img_path, label: (
        tf.image.resize(
            tf.image.decode_jpeg(tf.io.read_file(img_path), channels=3), [224, 224]
        ), tf.cast(label, tf.float32)
    ))

    dataset = dataset.batch(32).shuffle(buffer_size=len(df))

    return dataset

def load_client_data(client_data_paths):
    """
    Load data for each client from the provided CSV paths.
    Args:
        client_data_paths: List of paths to client CSVs.
    Returns:
        A dictionary where keys are client IDs and values are datasets.
    """
    client_data = {}

    for idx, path in enumerate(client_data_paths):
        # gather all CSV files in the directory because spark leaves them in separate .csvs
        all_csvs = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".csv")]
        combined_df = pd.concat((pd.read_csv(f) for f in all_csvs), ignore_index=True)

        feature_columns = combined_df[['Cardiomegaly', 'Lung Opacity', 'Edema', 'Consolidation', 'Pneumonia']].values
        image_paths = combined_df['Augmented_Path'].values
        
        # convert labels (features) to float32
        feature_columns = feature_columns.astype('float32')

        # convert pandas df into a tensorflow df
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, feature_columns))
        
        # resize the images to adjust to model
        dataset = dataset.map(lambda img_path, features: (
            tf.image.resize(
                tf.image.decode_jpeg(tf.io.read_file(img_path), channels=3), [224, 224]
            ), features
        ))

        dataset = dataset.batch(32).shuffle(buffer_size=len(combined_df))
        client_data[f"Client_{idx + 1}"] = dataset

    return client_data

def create_model():
    """
    Create a ResNet-based model architecture.
    """
    base_model = MobileNetV2(
        weights=None,  # initialize from scratch or use pre-trained weights - maybe?
        include_top=False,  # exclude the top layer
        input_shape=(224, 224, 3)
    )

    # add custom top layers
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(5, activation='sigmoid')
    ])

    return model

def model_fn():
    keras_model = tf.keras.Sequential([
        MobileNetV2(weights=None, include_top=False, input_shape=(224, 224, 3)),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(5, activation='sigmoid')
    ])
    return keras_utils.from_keras_model(
        keras_model=keras_model,
        input_spec=(
            tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 5), dtype=tf.float32),
        ),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=[tf.keras.metrics.BinaryAccuracy()]
    )


def initialize_federated_process():
    """
    Create a federated learning process using TFF.
    """
    iterative_process = tff.learning.algorithms.build_weighted_fed_avg(
        model_fn=model_fn,
        client_optimizer_fn=build_sgdm(learning_rate=0.02),
        server_optimizer_fn=build_sgdm(learning_rate=1.0)
    )
    return iterative_process

def train_federated_model(iterative_process, client_datasets, rounds=1):
    """
    Train the federated model across multiple rounds.
    Args:
        iterative_process: The TFF iterative process.
        client_datasets: Dictionary of client datasets.
        rounds: Number of training rounds.
    Returns:
        Final server state.
    """
    state = iterative_process.initialize()

    for round_num in range(1, rounds + 1):
        client_data = [client_datasets[client_id] for client_id in client_datasets]
        state, metrics = iterative_process.next(state, client_data)
        print(f'Round {round_num}, Metrics: {metrics}')

    return state

def evaluate_global_model(global_model, evaluation_dataset):
    """
    Evaluate the global model on a new dataset.
    Args:
        global_model: The trained Keras global model.
        evaluation_dataset: The evaluation dataset.
    """
    # Compile the model with the same loss and metrics used in training
    global_model.compile(
        optimizer=tf.keras.optimizers.Adam(),  # You can use the optimizer of your choice
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=[tf.keras.metrics.BinaryAccuracy()]
    )
    
    # Evaluate the model
    results = global_model.evaluate(evaluation_dataset, verbose=1)
    print(f"Evaluation Results - Loss: {results[0]}, Accuracy: {results[1]}")


# load evaluation data
evaluation_data_path = "chexlocalize/CheXpert/val_labels.csv"
evaluation_dataset = load_evaluation_data(evaluation_data_path)

# load client data
client_data_paths = [
    "output/clients/Client_1_data.csv",
    "output/clients/Client_2_data.csv",
    "output/clients/Client_3_data.csv",
    "output/clients/Client_4_data.csv",
]
client_datasets = load_client_data(client_data_paths)

# initialize the federated learning process
iterative_process = initialize_federated_process()

# train models
final_state = train_federated_model(iterative_process, client_datasets)

# extract the global model weights from the final state
model_weights = iterative_process.get_model_weights(final_state)
trainable_weights = model_weights.trainable
non_trainable_weights = model_weights.non_trainable

keras_model = model_fn()._keras_model

keras_model.set_weights(trainable_weights + non_trainable_weights)

# save the global model
# keras_model = create_model()

# # keras_model.set_weights(global_weights)
# keras_model.set_weights(global_weights)

# Save the global model
keras_model.save('output/global_model.h5')

# Evaluate the global model
evaluate_global_model(keras_model, evaluation_dataset)