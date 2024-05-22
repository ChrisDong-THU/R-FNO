import torch
import torch.nn as nn

import sys
import os

filename = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(filename)

from model import RecFieldCNN, RecFieldViT, FNO2d
from model.rfno import GRU2d

# model = GRU2d().cuda()
model = nn.GRU(input_size=32, hidden_size=32, num_layers=3, batch_first=True).cuda() # input_size为输入每个数据本身的大小

# 创建模拟输入数据
input_tensor = torch.randn(8, 4, 32) # [batch_size, seq_len, input_size] seq_len为序列长度
hidden_state = torch.randn(3, 8, 32) # [num_layers, batch_size, hidden_size]

input_tensor = input_tensor.cuda()
hidden_state = hidden_state.cuda()

# 执行前向传播
output = model(input_tensor, hidden_state)

# 检查输出
print(output[0].shape, output[1].shape)