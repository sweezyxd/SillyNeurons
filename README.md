# SillyNeurons

sillyneuron is a lightweight Python module designed to provide a simple implementation of a feedforward neural network for educational purposes. It offers basic functionalities for initializing weights and biases, performing forward propagation with ReLU, sigmoid and Softmax activations, and implementing backpropagation for training a neural network.

# Usage

Here's a straightforward illustration of training the model using the MNIST dataset (Same code can be found in [train.py](https://github.com/sweezyxd/SillyNeurons/blob/main/train.py)).
```
import sillyneuron as sn
import numpy as np
import pandas as pd

# Loading data
data = pd.read_csv("mnist_train.csv")
data = np.array(data)
x, y = [], []
for arr in data:
    x.append(sn.snf.normalize(arr[1:]))
    y.append(sn.snf.one_hot(arr[0]))

# Coding the model
model = sn.Model()

model.layer(16, .00001, "ReLU")
model.layer(16, .0001, "ReLU")
model.layer(10, .0001, "softmax")

model.train(x, y, 50, "ModelName")
```

# Functions
### `layer(n, rate, activation)`

**Description:** Adds a new layer to the neural network with the specified number of neurons, learning rate, and activation function.

**Arguments:**
- `n` (int): Number of neurons in the new layer.
- `rate` (float): Learning rate for weight and bias updates during training.
- `activation` (str): Activation function for the layer (options: "ReLU", "softmax").

### `train(X, Y, epochs, save=None)`

**Description:** Trains the neural network model with the given input data and target labels using backpropagation.

**Arguments:**
- `X` (numpy array): Input data for training the model (NOTE: the input given should be a 1D array).
- `Y` (numpy array): Target labels corresponding to the input data (NOTE: its better to have the output data as one hot encoded arrays).
- `epochs` (int): The number of training epochs.
- `save` (str, optional): If provided, the trained model will be saved for future use.

### `test(X, Y, epochs, save=None)`

**Description:** Trains the neural network model with the given input data and target labels using backpropagation.

**Arguments:**
- `X` (numpy array): Input data for training the model.
- `Y` (numpy array): Target labels corresponding to the input data.
- `epochs` (int): The number of training epochs.
- `save` (str, optional): If provided, the trained model will be saved for future use.

### `load(model_path)`

**Description:** Loads the entire model parameters (weights biases, learning rates, numbers of neurons and each layer's activation function).

**Argument:**
- `model_path` (str): The name of the saved model to load.

### `load_wb(model_path)`

**Description:** Loads only the weights and biases (layers should be rewritten in the code and have the same number of layers and neurons as the loaded weights and biases).

**Argument:**
- `model_path` (str): The name of the saved model to load.
