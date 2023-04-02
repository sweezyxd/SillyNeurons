import numpy as np


# Initializing weights and biases.
def Initialize(a1, a2):
    w = np.random.rand(int(a2), int(a1)) - 0.5
    b = np.random.random() - 0.5
    return w, b


# Applying weights and biases to the neurons, then applying an activation function.
def layer(a, w, b, activation):
    result = 'error'
    if activation == "ReLU":
        result = ReLU(np.dot(w, a) + b)
    if activation == "softmax":
        result = softmax(np.dot(w, a) + b)
    return result


# Normalizing data to get values between 0 and 1.
def normalize(inp):
    return inp / np.max(inp)


def ReLU(a):
    return np.maximum(0, a)


def deriv_ReLU(a):
    return a > 0


def softmax(a):
    a = a - np.max(a)
    return np.exp(a) / np.sum(np.exp(a))


# Manually one hot encoding the expected output (don't ask why).
def one_hot(Y):
    array = np.array([])
    for n in range(10):
        if n != Y:
            array = np.append(array, 0)
        else:
            array = np.append(array, 1)
    return array


# shulalulalu
def multiply(a, a2):
    result = []
    for val in a:
        result.append(val * a2)
    return np.array(result)


# Updating the weights and biases using back-propagation.
def backprop(W1, B1, W2, B2, A2, A1, X, Y, r):
    dZ2 = A2 - Y
    dW2 = multiply(dZ2, A1)
    dB2 = np.sum(dZ2)
    dZ = np.dot(W2.T, dZ2) * deriv_ReLU(A1)
    dW = multiply(dZ, X)
    dB = np.sum(dZ)
    return W1 - r * dW, B1 - r * dB, W2 - r * dW2, B2 - r * dB2


def update_softmax(A2, A1, Y):
    dZ = A2 - Y
    dW = multiply(dZ, A1)
    dB = np.sum(dZ)
    return dW, dB, dZ


def update_ReLU(W2, A1, A2, Z):
    dZ = np.dot(W2.T, Z) * deriv_ReLU(A2)
    dW = multiply(dZ, A1)
    dB = np.sum(dZ)
    return dW, dB, dZ


def sub(arr1, arr2):
    for n in range(len(arr1)):
        arr1[n] = arr1[n] - arr2[n]
    return arr1