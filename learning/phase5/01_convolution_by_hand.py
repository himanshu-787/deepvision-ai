"""Phase 5.1 - convolution and pooling from first principles."""

import numpy as np
import torch
import torch.nn.functional as F

# A 5x5 "image": dark on the left, bright on the right (a vertical edge)
img = np.array([[0, 0, 0, 9, 9]] * 5, dtype=float)
kernel = np.array([[-1, 0, 1]] * 3, dtype=float)       # vertical-edge detector


def conv2d_manual(image: np.ndarray, k: np.ndarray) -> np.ndarray:
    kh, kw = k.shape
    oh, ow = image.shape[0] - kh + 1, image.shape[1] - kw + 1   # stride 1, no padding
    out = np.zeros((oh, ow))
    for i in range(oh):
        for j in range(ow):
            out[i, j] = np.sum(image[i:i + kh, j:j + kw] * k)
    return out


manual = conv2d_manual(img, kernel)
print("manual convolution:\n", manual)        # each row: [0, 27, 27]

# PyTorch needs (N, C, H, W) shaped tensors
t_img = torch.tensor(img).reshape(1, 1, 5, 5)
t_ker = torch.tensor(kernel).reshape(1, 1, 3, 3)
torch_out = F.conv2d(t_img, t_ker)
print("PyTorch matches manual:", np.allclose(manual, torch_out.squeeze().numpy()))

# Output-size formula
for H, K, P, S in [(32, 3, 1, 1), (32, 3, 0, 1), (32, 3, 1, 2)]:
    print(f"H={H} K={K} P={P} S={S} -> {(H + 2 * P - K) // S + 1}")

# Max pooling 2x2
grid = torch.tensor([[1., 3., 2., 0.],
                     [4., 2., 1., 1.],
                     [0., 1., 5., 6.],
                     [2., 2., 7., 8.]]).reshape(1, 1, 4, 4)
print("max pool 2x2:\n", F.max_pool2d(grid, 2).squeeze())   # [[4, 2], [2, 8]]
