import torchLinearViz
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# モデル定義
model = SimpleModel()
input_tensor = torch.randn(1, 10)

torchLinearViz.extract_graph(model, input_tensor)


torchLinearViz.start(browser=True)
