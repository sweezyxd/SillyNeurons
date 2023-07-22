import sillyneuron as sn
import numpy as np
import pandas as pd

# Learns to draw

model = sn.Model()

data = pd.read_csv("mnist_train.csv")
data = np.array(data)
x, y = [], []
for arr in data:
    x.append(sn.snf.normalize(arr[1:]))
    y.append(sn.snf.one_hot(arr[0]))

model.load_wb("best2")

model.layer(32, .00001, "ReLU")
model.layer(32, .0001, "ReLU")
model.layer(10, .0001, "softmax")

model.train(x, y, 50, "best2")

