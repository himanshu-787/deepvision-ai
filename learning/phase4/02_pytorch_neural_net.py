"""Phase 4.2 - the same 2-16-1 network, with autograd doing the backprop."""

import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

torch.manual_seed(42)

X, y = make_moons(n_samples=600, noise=0.2, random_state=42)
y = y.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
mu, sigma = X_train.mean(axis=0), X_train.std(axis=0)
X_train, X_test = (X_train - mu) / sigma, (X_test - mu) / sigma

to_t = lambda a: torch.tensor(a, dtype=torch.float32)
X_train, X_test, y_train, y_test = map(to_t, (X_train, X_test, y_train, y_test))

model = nn.Sequential(
    nn.Linear(2, 16),   # (n,2)  -> (n,16)
    nn.Tanh(),
    nn.Linear(16, 1),   # (n,16) -> (n,1)  raw logits (no sigmoid here)
)
criterion = nn.BCEWithLogitsLoss()   # sigmoid + BCE fused: numerically safer
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

for epoch in range(2000):
    logits = model(X_train)
    loss = criterion(logits, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 400 == 0 or epoch == 1999:
        print(f"epoch {epoch:4d} | loss {loss.item():.4f}")


@torch.no_grad()
def accuracy(X: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    return ((torch.sigmoid(model(X)) > 0.5).float() == y).float().mean().item()


print(f"train accuracy: {accuracy(X_train, y_train):.3f} | test accuracy: {accuracy(X_test, y_test):.3f}")
