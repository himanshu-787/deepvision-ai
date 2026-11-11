"""Phase 4.1 - a 2-16-1 neural network with hand-written backpropagation."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(42)

# ---------- Data ----------
X, y = make_moons(n_samples=600, noise=0.2, random_state=42)
y = y.reshape(-1, 1).astype(float)                       # (600, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
mu, sigma = X_train.mean(axis=0), X_train.std(axis=0)    # statistics from TRAIN only
X_train, X_test = (X_train - mu) / sigma, (X_test - mu) / sigma

# ---------- Parameters ----------
H = 16
params = {
    "W1": rng.standard_normal((2, H)) * 0.5, "b1": np.zeros((1, H)),
    "W2": rng.standard_normal((H, 1)) * 0.5, "b2": np.zeros((1, 1)),
}


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def forward(p: dict, X: np.ndarray):
    z1 = X @ p["W1"] + p["b1"]       # (n,2)@(2,16) -> (n,16)
    a1 = np.tanh(z1)
    z2 = a1 @ p["W2"] + p["b2"]      # (n,16)@(16,1) -> (n,1)
    a2 = sigmoid(z2)
    return a1, a2


def bce_loss(a2: np.ndarray, y: np.ndarray) -> float:
    eps = 1e-12
    return float(-np.mean(y * np.log(a2 + eps) + (1 - y) * np.log(1 - a2 + eps)))


def backward(p: dict, X: np.ndarray, y: np.ndarray, a1: np.ndarray, a2: np.ndarray) -> dict:
    n = X.shape[0]
    dz2 = (a2 - y) / n
    dW2 = a1.T @ dz2
    db2 = dz2.sum(axis=0, keepdims=True)
    da1 = dz2 @ p["W2"].T
    dz1 = da1 * (1 - a1**2)
    dW1 = X.T @ dz1
    db1 = dz1.sum(axis=0, keepdims=True)
    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}


# ---------- Gradient check: is my backprop correct? ----------
a1, a2 = forward(params, X_train)
grads = backward(params, X_train, y_train, a1, a2)
eps = 1e-5
original = params["W1"][0, 0]
params["W1"][0, 0] = original + eps
loss_plus = bce_loss(forward(params, X_train)[1], y_train)
params["W1"][0, 0] = original - eps
loss_minus = bce_loss(forward(params, X_train)[1], y_train)
params["W1"][0, 0] = original
numeric = (loss_plus - loss_minus) / (2 * eps)
print(f"gradient check dL/dW1[0,0]: backprop={grads['W1'][0, 0]:.8f} numeric={numeric:.8f}")

# ---------- Training ----------
lr, epochs = 0.5, 2000
history = []
for epoch in range(epochs):
    a1, a2 = forward(params, X_train)                 # forward
    loss = bce_loss(a2, y_train)                      # loss
    grads = backward(params, X_train, y_train, a1, a2)  # backward
    for k in params:                                  # update (gradient descent)
        params[k] -= lr * grads[k]
    history.append(loss)
    if epoch % 400 == 0 or epoch == epochs - 1:
        print(f"epoch {epoch:4d} | loss {loss:.4f}")

# ---------- Evaluation ----------
def accuracy(X: np.ndarray, y: np.ndarray) -> float:
    return float(((forward(params, X)[1] > 0.5) == y).mean())

print(f"train accuracy: {accuracy(X_train, y_train):.3f} | test accuracy: {accuracy(X_test, y_test):.3f}")

# ---------- Plots ----------
out = Path("artifacts/plots")
out.mkdir(parents=True, exist_ok=True)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(history); axes[0].set_title("Training loss (NumPy)")
axes[0].set_xlabel("epoch"); axes[0].set_ylabel("BCE loss")
gx, gy = np.meshgrid(np.linspace(-2.5, 2.5, 200), np.linspace(-2.5, 2.5, 200))
grid = np.c_[gx.ravel(), gy.ravel()]
axes[1].contourf(gx, gy, forward(params, grid)[1].reshape(gx.shape), levels=20, cmap="RdBu", alpha=0.6)
axes[1].scatter(X_test[:, 0], X_test[:, 1], c=y_test.ravel(), cmap="RdBu", edgecolors="k", s=18)
axes[1].set_title("Decision boundary on test data")
plt.tight_layout()
plt.savefig(out / "phase4_numpy_net.png", dpi=120)
plt.close()
print("saved artifacts/plots/phase4_numpy_net.png")
