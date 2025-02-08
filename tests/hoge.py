import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 10)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        # x = F.relu(x)
        x = self.relu1(x)
        x = self.fc2(x)
        # x = F.relu(x)
        x = self.relu2(x)
        return x

'''model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 10)
)'''

# モデルの作成
model = SimpleModel()

# レイヤー間の接続を保存するための辞書
edges = []
layer_names = {}

# フォワードフックを使ってレイヤー間の接続を記録
def hook_fn(module, input, output):
    layer_name = layer_names[module]  # レイヤー名を取得
    if isinstance(output, torch.Tensor):
        output_id = id(output)  # 出力テンソルの ID を取得
        for inp in input:
            if isinstance(inp, torch.Tensor):
                input_id = id(inp)  # 入力テンソルの ID を取得
                edges.append((input_id, output_id, layer_name))  # (入力ID, 出力ID, レイヤー名) を保存

# 各レイヤーにフックを登録
for name, layer in model.named_modules():
    print('name :', name)
    if isinstance(layer, nn.Linear):  # 全結合層のみ
        layer_names[layer] = name  # レイヤー名を保存
        layer.register_forward_hook(hook_fn)  # フックを登録
        print(f"Layer: {name}, In Features: {layer.in_features}, Out Features: {layer.out_features}")

# ダミー入力を流してフックを発火させる
input_tensor = torch.randn(1, 10)
output = model(input_tensor)

# エッジ情報を整理
graph_edges = []
tensor_to_layer = {}

for (input_id, output_id, layer_name) in edges:
    tensor_to_layer[output_id] = layer_name  # 出力テンソルとレイヤー名を紐付け
    if input_id in tensor_to_layer:
        source_layer = tensor_to_layer[input_id]
        target_layer = layer_name
        graph_edges.append((source_layer, target_layer))

# **📌 グラフのエッジ情報を出力**
print("Graph Edges (Layer Connections):")
for edge in graph_edges:
    print(f"{edge[0]} → {edge[1]}")

