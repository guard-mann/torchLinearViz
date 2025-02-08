import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import ssl
from torchLinearViz import TorchLinearViz

ssl._create_default_https_context = ssl._create_unverified_context

# 🔹 1. データの前処理
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# 🔹 2. データセットの読み込み
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# 🔹 3. MLP（全結合ニューラルネットワーク）モデルの定義
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),  # 画像(28x28) → 1次元 (784)
            nn.Linear(28*28, 5),  # 入力 784 → 隠れ層 256 
#            nn.ReLU(),
            nn.Linear(5, 5),  # 隠れ層 256 → 128
#            nn.ReLU(),
            nn.Linear(5, 10)  # 出力 10クラス
        )

    def forward(self, x):
        return self.model(x)

# 🔹 4. モデルの作成
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLP().to(device)

# 🔹 5. 損失関数と最適化手法
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# torchLinearViz初期化
torchlinearviz = TorchLinearViz(model)

# 🔹 6. 学習
epochs = 100
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    torchlinearviz.update(model, images)

#    for name, param in model.named_parameters():
#        print(name, param.data.mean().item())  # 平均値を表示
#        print(name, param)

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}")

torchlinearviz.end()
