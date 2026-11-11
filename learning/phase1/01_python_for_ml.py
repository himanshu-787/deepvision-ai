"""Phase 1.1 - Python essentials used throughout the project.

Run from project root:
    python learning/phase1/01_python_for_ml.py
"""

import json
import tempfile
from pathlib import Path


# ---------- Functions (with type hints) ----------
def accuracy(correct: int, total: int) -> float:
    """Fraction of predictions that were right."""
    if total == 0:
        raise ValueError("total must be greater than 0")
    return correct / total


# ---------- Lists, tuples, dictionaries ----------
losses = [2.30, 1.90, 1.50, 1.20]          # list: ordered, changeable
squared = [loss**2 for loss in losses]      # list comprehension
image_size = (32, 32)                       # tuple: ordered, NOT changeable
config = {"lr": 0.01, "epochs": 5, "batch_size": 64}  # dict: key -> value

# enumerate gives index + value, zip walks lists in parallel
for epoch, loss in enumerate(losses, start=1):
    print(f"epoch {epoch}: loss={loss}")
for name, size in zip(["conv1", "conv2"], [32, 64]):
    print(name, size)


# ---------- Classes (object-oriented programming) ----------
class RunningAverage:
    """Tracks the average of numbers seen so far (we reuse this idea in training)."""

    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += value * n
        self.count += n

    @property
    def average(self) -> float:
        return self.total / self.count if self.count else 0.0


meter = RunningAverage()
for loss in losses:
    meter.update(loss)
print("average loss:", meter.average)


# ---------- File handling (JSON) ----------
with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "config.json"
    path.write_text(json.dumps(config, indent=2))
    loaded = json.loads(path.read_text())
    print("loaded config:", loaded)


# ---------- Exception handling ----------
try:
    accuracy(5, 0)
except ValueError as err:
    print("caught error:", err)

print("accuracy(90, 100) =", accuracy(90, 100))
