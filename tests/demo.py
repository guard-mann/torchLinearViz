from torchLinearViz import TorchLinearViz
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(3, 5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return x

# モデル定義
model = SimpleModel()
input_tensor = torch.randn(1, 3)  # 1 サンプル, 10 次元入力
target = torch.randn(1, 5)  # 1 サンプル, 10 次元の正解データ (回帰用)
optimizer = optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()  # 回帰タスク用の損失関数


# torchLinearViz初期化
torchlinearviz = TorchLinearViz(model)
torchlinearviz.start(browser=True)

# 学習
epochs = 10
for epoch in range(epochs):
    optimizer.zero_grad()
    time.sleep(5)
    output = model(input_tensor)
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()

    torchlinearviz.update(model)

torchlinearviz.end()
