from torchLinearViz import TorchLinearViz
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(3, 100)
        self.fc2 = nn.Linear(100, 200)
        self.fc3 = nn.Linear(200, 200)
        self.fc4 = nn.Linear(200, 200)
        self.fc5 = nn.Linear(200, 200)
        self.fc6 = nn.Linear(200, 200)
        self.fc7 = nn.Linear(200, 10)



    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.fc6(x)
        x = self.fc7(x)
        return x

# モデル定義
model = SimpleModel()
input_tensor = torch.randn(100, 3)  # 1 サンプル, 3次元入力
target = torch.randn(100, 10)  # 10サンプル, 2次元の正解データ (回帰用)
optimizer = optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()  # 回帰タスク用の損失関数


# torchLinearViz初期化
torchlinearviz = TorchLinearViz(model)
torchlinearviz.start(browser=True)

# 学習
epochs = 2
for epoch in range(epochs):
    optimizer.zero_grad()
    output = model(input_tensor)
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()


    print(f"Epoch {epoch + 1} - Weight Check:", loss)
    for name, param in model.named_parameters():
        print(name, param.data.mean().item())  # 平均値を表示
        print(name, param)


    torchlinearviz.update(model, input_tensor)

torchlinearviz.end()
