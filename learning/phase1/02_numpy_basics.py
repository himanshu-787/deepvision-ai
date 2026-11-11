"""Phase 1.2 - NumPy: the language of neural network math."""

import numpy as np

# ---- Scalar, vector, matrix, tensor ----
scalar = np.array(5.0)
vector = np.array([1.0, 2.0, 3.0])
matrix = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
print("shapes:", scalar.shape, vector.shape, matrix.shape)  # () (3,) (3, 2)

# An image is a 3-D array: height x width x channels
fake_image = np.random.randint(0, 256, size=(32, 32, 3), dtype=np.uint8)
print("image shape (H, W, C):", fake_image.shape)

# ---- One artificial neuron ----
x = np.array([1.0, 2.0, 3.0])        # 3 input features
w = np.array([0.2, -0.5, 0.1])       # 3 weights
b = 0.4                              # bias
z = np.dot(x, w) + b                 # 0.2 - 1.0 + 0.3 + 0.4 = -0.1
a = max(0.0, z)                      # ReLU activation: negatives become 0
print(f"neuron: z={z:.2f}, after ReLU={a:.2f}")

# ---- A whole layer = matrix multiplication ----
X = np.random.randn(4, 3)            # batch of 4 samples, 3 features each
W = np.random.randn(3, 2)            # layer with 3 inputs -> 2 neurons
bias = np.array([0.5, -0.5])         # shape (2,)
out = X @ W + bias                   # (4,3) @ (3,2) -> (4,2); bias broadcasts
print("layer output shape:", out.shape)

# ---- Broadcasting ----
A = np.ones((2, 3))
v = np.array([10, 20, 30])
print("broadcast add:\n", A + v)     # v is stretched across both rows

# ---- Mean, variance, standardization (used for image normalization) ----
data = np.array([[1.0, 100.0], [2.0, 200.0], [3.0, 300.0]])
mean = data.mean(axis=0)             # axis=0: down the rows, per column
std = data.std(axis=0)
print("mean:", mean, "std:", std)
print("standardized:\n", (data - mean) / std)  # each column: mean 0, std 1

# ---- Reshape and flatten ----
flat = fake_image.reshape(-1)        # -1 means "figure this dimension out"
print("flattened length:", flat.size)  # 32*32*3 = 3072
