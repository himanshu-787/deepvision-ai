"""Phase 1.4 - softmax and cross-entropy by hand."""

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)      # subtract max: prevents overflow, same result
    exps = np.exp(shifted)
    return exps / exps.sum()


def cross_entropy(probs: np.ndarray, true_class: int) -> float:
    return -np.log(probs[true_class] + 1e-12)   # epsilon avoids log(0)


logits = np.array([2.0, 1.0, 0.1])
probs = softmax(logits)
print("probabilities:", probs.round(3), "sum =", probs.sum().round(3))  # ~[0.659 0.242 0.099]

print("loss if true class is 0 (model right):", round(cross_entropy(probs, 0), 3))
print("loss if true class is 2 (model wrong):", round(cross_entropy(probs, 2), 3))

confident_wrong = softmax(np.array([8.0, 0.0, 0.0]))
print("confidently wrong loss:", round(cross_entropy(confident_wrong, 2), 3))

# Statistics refresher
scores = np.array([70, 80, 90, 100])
print("mean:", scores.mean(), "variance:", scores.var(), "std:", scores.std().round(3))
