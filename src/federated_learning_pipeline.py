import tensorflow as tf
import tensorflow_federated as tff
from tensorflow.keras.applications import ResNet50

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
        data = tf.data.experimental.make_csv_dataset(
            path, batch_size=32, shuffle=True, num_epochs=1, ignore_errors=True
        )
        client_data[f"Client_{idx + 1}"] = data
    return client_data

def create_model():
    """
    Create a ResNet-based model architecture.
    """
    base_model = ResNet50(
        weights=None,  # initialize from scratch or use pre-trained weights - maybe?
        include_top=False,  # exclude the top layer
        input_shape=(224, 224, 3)
    )

    # add custom top layers
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    return mode

def model_fn():
    """
    Create the model function for TFF.
    """
    keras_model = create_model()
    return tff.learning.from_keras_model(
        keras_model,
        input_spec={
            'features': tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),
            'label': tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
        },
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy()]
    )

def initialize_federated_process():
    """
    Create a federated learning process using TFF.
    """
    iterative_process = tff.learning.build_federated_averaging_process(
        model_fn,
        client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.02),
        server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0)
    )
    return iterative_process

def train_federated_model(iterative_process, client_datasets, rounds=10):
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

# save the global model
keras_model = create_model()
keras_model.set_weights(tff.learning.state_with_model_weights(final_state)['model'])
keras_model.save('output/global_model.h5')