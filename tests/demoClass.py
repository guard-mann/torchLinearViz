from torchLinearViz import TorchLinearViz
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torch.optim as optim
import time

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(3, 5)
#        self.fc2 = nn.Linear(20, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
#        x = self.fc2(x)
#        x = F.relu(x)
        return x

# モデル定義
model = SimpleModel()
input_tensor = torch.randn(1, 3)  # 1 サンプル, 10 次元入力
target = torch.randn(1, 5)  # 1 サンプル, 10 次元の正解データ (回帰用)
optimizer = optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()  # 回帰タスク用の損失関数


torchlinearviz = TorchLinearViz(model)
torchlinearviz.start(browser=True)

epochs = 10
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")

    optimizer.zero_grad()
    time.sleep(5)
    output = model(input_tensor)
    loss = loss_fn(output, target)  # ロス計算
    print(loss)
    loss.backward()
    optimizer.step()  # フックがここで呼び出される


    for name, param in model.named_parameters():
        print('TEST :', name, param.data.mean().item())  # 平均値を表示
    if epoch > 4:
        with torch.no_grad():
            model.fc1.weight.fill_(1.0)  # `fc2` の全重みを `1` に変更
    torchlinearviz.update(model)

# 終了
torchlinearviz.end()
