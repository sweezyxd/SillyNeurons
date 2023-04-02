import sillyneuron as sn
import numpy as np
import pandas as pd


model = sn.Model()

data = pd.read_csv("fashion_mnist_train.csv")
data = np.array(data)
x, y = [], []
for arr in data:
    x.append(arr[1:])
    y.append(arr[0])

model.input = x

model.layer(10, .001, "ReLU")
model.layer(10, .001, "ReLU")
model.layer(10, .01, "softmax")

model.train(y, epochs=50, save="fashion_trained")
