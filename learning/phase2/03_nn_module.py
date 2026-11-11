"""Phase 2.3 - nn.Module, parameters vs buffers, and a real training loop."""

import torch
import torch.nn as nn

torch.manual_seed(0)

# ---- Data: learn y = 3x + 2 (same task as Phase 1, now in PyTorch) ----
x = torch.linspace(-1, 1, 100).unsqueeze(1)          # shape (100, 1)
y = 3 * x + 2 + 0.1 * torch.randn_like(x)

model = nn.Linear(in_features=1, out_features=1)     # learns w and b
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(200):
    pred = model(x)                  # 1. forward
    loss = criterion(pred, y)        # 2. loss
    optimizer.zero_grad()            # 3. clear old gradients
    loss.backward()                  # 4. backward (compute gradients)
    optimizer.step()                 # 5. update parameters
    if epoch % 50 == 0:
        print(f"epoch {epoch:3d} | loss {loss.item():.4f}")

print("learned w:", model.weight.item(), "b:", model.bias.item())


# ---- A custom Module ----
class TinyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(8, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))       # (B,4) -> (B,8) -> (B,3)


net = TinyNet()
print("\noutput shape:", net(torch.randn(5, 4)).shape)      # (5, 3)
for name, p in net.named_parameters():
    print(f"{name:12s} shape={tuple(p.shape)}")
print("total params:", sum(p.numel() for p in net.parameters()))  # 4*8+8 + 8*3+3 = 67

# ---- Parameters vs buffers ----
bn = nn.BatchNorm1d(4)
print("\nBatchNorm parameters:", [n for n, _ in bn.named_parameters()])
print("BatchNorm buffers   :", [n for n, _ in bn.named_buffers()])

# ---- train() vs eval() ----
net.train(); print("training mode:", net.training)
net.eval();  print("eval mode:", net.training)
