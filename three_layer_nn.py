import numpy as np
import matplotlib.pyplot as plt

# 3 layer NN from scratch, no keras/torch
# input -> hidden1 (relu) -> hidden2 (relu) -> output (softmax)

def relu(z):
    return np.maximum(0, z)

def relu_deriv(z):
    return (z > 0).astype(float)

def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)

class NN:
    def __init__(self, n_in, n_h1, n_h2, n_out):
        np.random.seed(42)
        self.W1 = np.random.randn(n_in, n_h1) * np.sqrt(2/n_in)
        self.b1 = np.zeros((1, n_h1))
        self.W2 = np.random.randn(n_h1, n_h2) * np.sqrt(2/n_h1)
        self.b2 = np.zeros((1, n_h2))
        self.W3 = np.random.randn(n_h2, n_out) * np.sqrt(2/n_h2)
        self.b3 = np.zeros((1, n_out))

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = relu(self.z2)
        self.z3 = self.a2 @ self.W3 + self.b3
        self.a3 = softmax(self.z3)
        return self.a3

    def loss(self, pred, y):
        m = y.shape[0]
        return -np.sum(y * np.log(pred + 1e-9)) / m

    def backward(self, X, y, lr):
        m = X.shape[0]

        dz3 = self.a3 - y  # softmax + cross entropy grad
        dW3 = self.a2.T @ dz3 / m
        db3 = np.sum(dz3, axis=0, keepdims=True) / m

        da2 = dz3 @ self.W3.T
        dz2 = da2 * relu_deriv(self.z2)
        dW2 = self.a1.T @ dz2 / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        da1 = dz2 @ self.W2.T
        dz1 = da1 * relu_deriv(self.z1)
        dW1 = X.T @ dz1 / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        self.W3 -= lr * dW3
        self.b3 -= lr * db3
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

    def acc(self, X, y):
        return np.mean(self.predict(X) == np.argmax(y, axis=1))


def one_hot(y, k):
    oh = np.zeros((y.shape[0], k))
    oh[np.arange(y.shape[0]), y] = 1
    return oh


# Generate spiral dataset for multiclass classification
def spiral_data(n=150, classes=3):
    X = np.zeros((n*classes, 2))
    y = np.zeros(n*classes, dtype=int)
    for j in range(classes):
        ix = range(n*j, n*(j+1))
        r = np.linspace(0, 1, n)
        t = np.linspace(j*4, (j+1)*4, n) + np.random.randn(n)*0.2
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        y[ix] = j
    return X, y


if __name__ == "__main__":
    X, y = spiral_data()
    Y = one_hot(y, 3)
    X = (X - X.mean(0)) / X.std(0)  # normalize, helps a lot with convergence

    net = NN(2, 16, 16, 3)

    epochs = 2000
    lr = 0.05
    batch = 64
    losses = []

    for ep in range(epochs):
        perm = np.random.permutation(len(X))
        Xs, Ys = X[perm], Y[perm]
        for i in range(0, len(X), batch):
            xb = Xs[i:i+batch]
            yb = Ys[i:i+batch]
            net.forward(xb)
            net.backward(xb, yb, lr)

        pred = net.forward(X)
        l = net.loss(pred, Y)
        losses.append(l)
        if ep % 200 == 0:
            print(ep, l, net.acc(X, Y))

    final_acc = net.acc(X, Y)
    print(f"Final Accuracy = {final_acc*100:.1f}%")

    # quick plots so i can see if it actually worked
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].plot(losses)
    ax[0].set_title("Training Loss")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Cross Entropy Loss")
    ax[0].grid(True, alpha=0.3)

    xx, yy = np.meshgrid(np.linspace(X[:,0].min()-.5, X[:,0].max()+.5, 300),
                          np.linspace(X[:,1].min()-.5, X[:,1].max()+.5, 300))
    Z = net.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax[1].contourf(xx, yy, Z, alpha=0.3)
    ax[1].scatter(X[:,0], X[:,1], c=y, edgecolor='k', s=25)
    ax[1].set_title("Decision Boundary")
    ax[1].set_xlabel("Feature 1")
    ax[1].set_ylabel("Feature 2")

    fig.suptitle(f"Training Accuracy = {final_acc*100:.1f}%")
    plt.savefig("/mnt/user-data/outputs/nn_results.png", dpi=150)
    print("saved plot")
