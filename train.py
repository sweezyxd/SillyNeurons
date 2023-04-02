import sillyneuron as sn
import numpy as np
import pandas as pd


model = sn.Model()

data = pd.read_csv("fashion-mnist_test.csv")
data = np.array(data)
x, y = [], []
for arr in data:
    x.append(arr[1:])
    y.append(arr[0])

model.load('sn.models/fashion_trained32')

model.layer(32, .01, "ReLU")
model.layer(16, .01, "ReLU")
model.layer(10, .01, "softmax")

model.test(x, y)
