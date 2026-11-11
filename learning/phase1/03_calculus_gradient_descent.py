"""Phase 1.3 - derivatives, chain rule, gradient descent, linear regression."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ---------- 1. Numerical vs analytical derivative ----------
def f(w: float) -> float:
    return (w - 3.0) ** 2

def df(w: float) -> float:
    return 2.0 * (w - 3.0)

def numerical_derivative(func, w: float, eps: float = 1e-5) -> float:
    return (func(w + eps) - func(w - eps)) / (2 * eps)

print("analytical df(0):", df(0.0), "| numerical:", round(numerical_derivative(f, 0.0), 4))

# ---------- 2. Chain rule check ----------
# y = (3x + 1)^2  ->  inner g=3x+1, outer f=g^2  ->  dy/dx = 2*(3x+1)*3
chain = lambda x: (3 * x + 1) ** 2
x0 = 2.0
print("chain rule:", 2 * (3 * x0 + 1) * 3, "| numerical:", round(numerical_derivative(chain, x0), 3))

# ---------- 3. Gradient descent on f(w) ----------
w, lr = 0.0, 0.1
for step in range(25):
    w = w - lr * df(w)
print(f"gradient descent found w = {w:.4f} (true minimum is 3)")

# ---------- 4. Linear regression with hand-written gradients ----------
rng = np.random.default_rng(42)
x = rng.uniform(-1, 1, size=100)
y = 3.0 * x + 2.0 + rng.normal(0, 0.1, size=100)   # true w=3, b=2 plus noise

w, b, lr = 0.0, 0.0, 0.1
loss_history = []
for epoch in range(200):
    pred = w * x + b                      # forward pass
    error = pred - y
    loss = np.mean(error**2)              # MSE loss
    dw = np.mean(2 * error * x)           # dLoss/dw via chain rule
    db = np.mean(2 * error)               # dLoss/db
    w -= lr * dw                          # update
    b -= lr * db
    loss_history.append(loss)

print(f"learned w={w:.3f}, b={b:.3f} (true values 3 and 2)")

out = Path("artifacts/plots")
out.mkdir(parents=True, exist_ok=True)
plt.plot(loss_history)
plt.xlabel("epoch"); plt.ylabel("MSE loss"); plt.title("Gradient descent")
plt.savefig(out / "phase1_gd_loss.png", dpi=120)
plt.close()
print("saved artifacts/plots/phase1_gd_loss.png")
