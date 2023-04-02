import os.path
import sillyneuronfunctions as snf
import numpy as np


class Model:
    def __init__(self):
        self.w, self.b, self.n = [], [], []
        self.activation = np.array([])
        self.nn_count = np.array([])
        self.n_count = 0
        self.rate = np.array([])

        self.input = None

    def layer(self, n, rate, activation):
        self.rate = np.append(self.rate, rate)
        self.n_count += 1
        self.nn_count = np.append(self.nn_count, n)
        self.activation = np.append(self.activation, activation)

    def load(self, model_path):
        self.w, self.b = np.load(model_path+"/weights.w.npy"), np.load(model_path+"/biases.b.npy")

    def train(self, Y, epochs, save=None):
        right, total = 0, 0
        self.nn_count = np.insert(self.nn_count, 0, len(self.input[0]))
        for epoch in range(epochs):
            # FORWARD PROPAGATION
            epoch += 1
            for e in range(len(Y)):  # len(self.input)
                y = Y[e]
                self.n = [snf.normalize(self.input[e])]
                # if there are no weights or biases, they will get automatically generated.
                if len(self.w) == 0:
                    self.w, self.b = [], []
                    for n in range(self.n_count):
                        w, b = snf.Initialize(self.nn_count[n], self.nn_count[n+1])
                        self.w.append(w)
                        self.b.append(b)

                # activation(a*w + b)
                for n in range(len(self.w)):
                    self.n.append(snf.layer(self.n[n], self.w[n], self.b[n], self.activation[n]))
                if y == list(self.n[-1]).index(np.max(self.n[-1])):
                    right += 1
                total += 1
                accuracy = right / total
                if total % 1000 == 0:
                    os.system("cls")
                    print("=======================================")
                    print("Accuracy:", accuracy, "|| Epochs:", epoch)
                    print("=======================================")

                # BACK PROPAGATION
                temp_w, temp_b = [], []
                for n in range(self.n_count):
                    n += 1
                    if self.activation[-n] == "softmax":
                        dw, db, dz = snf.update_softmax(self.n[-n], self.n[-n-1], snf.one_hot(y))
                    if self.activation[-n] == "ReLU":
                        dw, db, dz = snf.update_ReLU(self.w[-n+1], self.n[-n-1], self.n[-n], dz)
                    temp_b.append(db * self.rate[-n]), temp_w.append(dw * self.rate[-n])
                temp_w.reverse()
                temp_b.reverse()
                self.w = snf.sub(self.w, temp_w)
                self.b = snf.sub(self.b, temp_b)
                dz = None
        # Saving the weights and biases
        if save is not None:
            if not os.path.exists("sn.models"):
                os.mkdir("sn.models")
            if os.path.exists("sn.models/" + save):
                os.remove("sn.models/" + save + "/weights.w.npy")
                os.remove("sn.models/" + save + "/biases.b.npy")
                os.remove("sn.models/" + save + "/info.txt")
            else:
                os.mkdir("sn.models/" + save)
            np.save("sn.models/" + save + "/weights.w", self.w)
            np.save("sn.models/" + save + "biases.b", self.b)
            with open("sn.models/" + save + "info.txt", "w") as w:
                w.write("======================================\n")
                w.write("Accuracy: " + str(accuracy) + "\n")
                w.write("Layers: " + str(len(self.n_count) - 1) + " Hidden layers." + "\n")
                for n in range(len(self.n_count) - 1):
                    n += 1
                    w.write("Hidden layer:" + str(self.nn_count[n]) + " neurons, " + str(self.activation[n]) + " activation.\n")
                w.write("Output layer:" + str(self.nn_count[-1]) + " neurons, " + str(self.activation[-1]) + " activation.\n")
                w.write("======================================\n")
                w.close()
            print("Model saved.")
