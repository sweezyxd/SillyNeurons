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
        self.b = list(np.load("sn.models/"+model_path+"/biases.b.npy", allow_pickle=True))
        for n in range(int(open("sn.models/"+model_path+"/weights/weight_info.txt").read())):
            self.w.append(np.load("sn.models/"+model_path+"/weights/weight_"+str(n)+".w.npy", allow_pickle=True))

        self.rate = list(np.load("sn.models/" + model_path + "/layers/rate.npy", allow_pickle=True))
        self.nn_count = list(np.load("sn.models/" + model_path + "/layers/nn_count.npy", allow_pickle=True))
        self.activation = list(np.load("sn.models/" + model_path + "/layers/activation.npy", allow_pickle=True))
        self.n_count = int(open("sn.models/" + model_path + "/layers/n_count.txt").read())

    def load_wb(self, model_path):
        self.b = list(np.load("sn.models/"+model_path+"/biases.b.npy", allow_pickle=True))
        for n in range(int(open("sn.models/"+model_path+"/weights/weight_info.txt").read())):
            self.w.append(np.load("sn.models/"+model_path+"/weights/weight_"+str(n)+".w.npy", allow_pickle=True))

    def train(self, X, Y, epochs, save=None):
        right, total = 0, 0
        self.input = X
        self.nn_count = np.insert(self.nn_count, 0, len(self.input[0]))
        for epoch in range(epochs):
            epoch += 1
            for e in range(len(Y)):
                # FORWARD PROPAGATION
                y = Y[e]
                self.n = [self.input[e]]
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

                if list(y).index(np.max(y)) == list(self.n[-1]).index(np.max(self.n[-1])):
                    right += 1
                total += 1
                accuracy = right / total
                if total % 1000 == 0:
                    os.system("cls")
                    print("=======================================")
                    print("Accuracy:", accuracy, "|| Epochs:", epoch, "|| COST:", np.sum(np.square(self.n[-1] - y)))
                    print("=======================================")

                # BACK PROPAGATION
                temp_w, temp_b = [], []
                dz = 2 * (self.n[-1] - y)
                for n in range(self.n_count):
                    n += 1
                    if self.activation[-n] == "softmax":
                        dw, db, dz = snf.update_softmax(self.n[-n-1], dz)
                    if self.activation[-n] == "ReLU":
                        dw, db, dz = snf.update_ReLU(self.w[-n+1], self.n[-n-1], self.n[-n], dz)
                    if self.activation[-n] == "sigmoid":
                        if -n == -1:
                            dw, db, dz = snf.update_sigmoid_output(self.n[-n-1], dz)
                        else:
                            dw, db, dz = snf.update_sigmoid(self.w[-n + 1], self.n[-n - 1], self.n[-n], dz)
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
                try:
                    for file in os.listdir("sn.models/" + save + "/layers"):
                        os.remove("sn.models/" + save + "/layers/" + file)
                    for n in range(int(open("sn.models/" + save + "/weights/weight_info.txt").read())):
                        os.remove("sn.models/" + save + "/weights/weight_" + str(n) + ".w.npy")
                    os.remove("sn.models/" + save + "/biases.b.npy")
                except FileNotFoundError:
                    print("the model you wanted to save at is corrupted and cannot save the trained model into, please remove the model's folder from the sn.models folder.")
            else:
                os.mkdir("sn.models/" + save)
                os.mkdir("sn.models/" + save + "/weights")
                os.mkdir("sn.models/" + save + "/layers")
            for n in range(len(self.w)):
                np.save("sn.models/" + save + "/weights/weight_"+str(n)+".w.npy", self.w[n])
            np.save("sn.models/" + save + "/layers/rate.npy", self.rate)
            np.save("sn.models/" + save + "/layers/nn_count.npy", self.nn_count)
            np.save("sn.models/" + save + "/layers/activation.npy", self.activation)
            open("sn.models/" + save + "/layers/n_count.txt", 'w').write(str(self.n_count))
            open("sn.models/" + save + "/weights/weight_info.txt", 'w').write(str(len(self.w)))
            np.save("sn.models/" + save + "/biases.b", self.b)
            try:
                lines = open("sn.models/" + save + "/info.txt").readlines()
            except FileNotFoundError:
                lines = []
            with open("sn.models/" + save + "/info.txt", "w+") as w:
                epch = 0
                for line in lines:
                    if "Epochs: " in line:
                        epch = int(line.replace("Epochs: ", "").replace(".", ""))
                w.write("======================================\n")
                w.write("Accuracy: " + str(accuracy) + ".\n")
                w.write("Epochs: " + str(epch + epochs) + ".\n")
                w.write("Layers: " + str(self.n_count - 1) + " Hidden layers." + "\n")
                for n in range(self.n_count - 1):
                    n += 1
                    w.write("Hidden layer:" + str(self.nn_count[n]) + " neurons, " + str(self.rate[n-1]) + " learning rate, " + str(self.activation[n-1]) + " activation.\n")
                w.write("Output layer:" + str(self.nn_count[-1]) + " neurons, " + str(self.rate[-1]) + " learning rate, " + str(self.activation[-1]) + " activation.\n")
                w.write("======================================\n")
                w.close()
            print("Model saved.")

    def test(self, X, Y, auto=False):
        self.input = X
        self.nn_count = np.insert(self.nn_count, 0, len(self.input[0]))
        right, total = 0, 0
        # FORWARD PROPAGATION
        for e in range(len(Y)):
            y = Y[e]
            self.n = [self.input[e]]

            # activation(a*w + b)
            for n in range(len(self.w)):
                self.n.append(snf.layer(self.n[n], self.w[n], self.b[n], self.activation[n]))

            if not auto:
                os.system("cls")
                print("Predicted:", list(self.n[-1]).index(np.max(self.n[-1])), "|| Correct:", list(y).index(np.max(y)))
                input("press enter...")
            else:
                if list(y).index(np.max(y)) == list(self.n[-1]).index(np.max(self.n[-1])):
                    right += 1
                total += 1
                accuracy = right / total
                if total % 1000 == 0:
                    os.system("cls")
                    data_rgb = np.array([])
                    for x in self.n[-1]:
                        data_rgb = np.append(data_rgb, np.array([x, x, x]))
                    os.system("cls")
                    print("=======================================")
                    print("Accuracy:", accuracy, "|| Correct answers:", len(Y), "/", right)
                    print("=======================================")
        # Final results:
        os.system("cls")
        print("Results:")
        print("=======================================")
        print("Accuracy:", accuracy, "|| Correct answers:", len(Y), "/", right)
        print("=======================================")