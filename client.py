import flwr as fl
import tensorflow as tf
from tensorflow import keras
import sys

# Load and compile Keras model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

# Load dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(x_train, y_train, epochs=5)
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test)
        return loss, len(x_test), {"accuracy": accuracy}

# Start Flower client
def main():
    if len(sys.argv) != 2:
        print("abey bhidu run: python(python3) pythonfile.py <port no>")
        sys.exit(1)

    try:
        port = int(sys.argv[1])
    except ValueError:
        print("Error: Port must be an integer")
        sys.exit(1)

    # Create strategy and run server
    strategy_MINE = SaveModelStrategy()
    server_address = f"localhost:{port}"
    config = fl.server.ServerConfig(num_rounds=3)
    fl.server.start_server(
        server_address=server_address,
        config=config,
        grpc_max_message_length=1024*1024*1024,
        strategy=strategy_MINE
    )

if __name__ == "_main_":
    main()
