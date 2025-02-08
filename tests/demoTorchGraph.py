import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 10)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
#        x = F.relu(x)  # ← functional の ReLU
        x = self.relu2(x)  # ← nn.ReLU() の ReLU
        x = self.fc2(x)
#        x = F.relu(x)  # ← functional の ReLU
        x = self.relu1(x)  # ← nn.ReLU() の ReLU
        return x

# モデルの作成
model = SimpleModel()


modelSequential = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 10)
)

model = models.resnet18()

# レイヤー間の接続を保存するための辞書
edges = []
layer_names = {}
last_valid_layer = None  # 直前の有効なレイヤーを追跡
tensor_to_layer = {}  # テンソル ID からレイヤー名へのマッピング

# フォワードフックを使ってレイヤー間の接続を記録
def hook_fn(module, input, output):
    global last_valid_layer
    layer_name = layer_names[module]  # レイヤー名を取得
    
    if isinstance(output, torch.Tensor):
        output_id = id(output)  # 出力テンソルの ID を取得

        # 直前の有効なレイヤーが存在すればエッジを追加
        if last_valid_layer is not None:
            edges.append((last_valid_layer, layer_name))  # (前のレイヤー → 現在のレイヤー)
        
        # テンソル ID をレイヤー名と紐付ける
        tensor_to_layer[output_id] = layer_name
        last_valid_layer = layer_name  # 最後に処理したレイヤーを更新

# 各レイヤーにフックを登録
for name, layer in model.named_modules():
#    if isinstance(layer, (nn.Linear, nn.ReLU)):  # 追加したいレイヤー
    layer_names[layer] = name  # レイヤー名を保存
    layer.register_forward_hook(hook_fn)  # フックを登録
    if isinstance(layer, nn.Linear):  # 全結合層のみ
        print(f"---\nLayer: {name}\nType: {layer.__class__.__name__}\nIn Features: {layer.in_features}, Out Features: {layer.out_features}")
    else:
        print(f'---\nLayer: {name}\nType: {layer.__class__.__name__}')

# ダミー入力を流してフックを発火させる
input_tensor = torch.randn(1, 3, 224, 224)# (1, 10)
output = model(input_tensor)

# **📌 グラフのエッジ情報を出力**
print('\n\n\n*******************************************************')
print("Graph Edges (Layer Connections):")
for edge in edges:
    print(f"{edge[0]} → {edge[1]}")

