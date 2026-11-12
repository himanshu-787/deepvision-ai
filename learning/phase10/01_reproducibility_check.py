"""Phase 10 - does seeding make training repeatable? (tiny model, no dataset needed)"""

from typing import Optional

import torch
import torch.nn as nn

from deepvision.utils.seed import set_seed


def train_tiny(seed: Optional[int]) -> torch.Tensor:
    if seed is not None:
        set_seed(seed)
    x, y = torch.randn(256, 8), torch.randint(0, 3, (256,))
    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 3))
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    for _ in range(50):
        opt.zero_grad()
        nn.functional.cross_entropy(model(x), y).backward()
        opt.step()
    return torch.cat([p.detach().flatten() for p in model.parameters()])


a, b = train_tiny(42), train_tiny(42)
c = train_tiny(7)
d1, d2 = train_tiny(None), train_tiny(None)
print("same seed, run twice     -> identical weights:", torch.equal(a, b))
print("different seed           -> identical weights:", torch.equal(a, c))
print("no seeding, run twice    -> identical weights:", torch.equal(d1, d2))
