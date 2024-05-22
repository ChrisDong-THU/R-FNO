import torch
import torch.nn as nn

import sys
import os

filename = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(filename)

from model import RecFieldCNN, RecFieldViT, FNO2d
from model.rfno import GRU2d

model = GRU2d().cuda()

# 创建模拟输入数据
input_tensor = torch.randn(1, 128, 112, 192) # [batch_size, seq_len, input_size]
hidden_state = torch.randn(1, 128, 112, 192) # [num_layers, batch_size, hidden_size]

input_tensor = input_tensor.cuda()
hidden_state = hidden_state.cuda()

# 执行前向传播
output = model(input_tensor, hidden_state)

# 检查输出
print(output.shape)