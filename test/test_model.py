import torch

import sys
import os

filename = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(filename)

from model import RecFieldCNN, RecFieldViT, FNO2d

# model = RecFieldCNN(in_channels=2).cuda()
model = RecFieldViT(in_channels=1).cuda()
# model = FNO2d(modes1=32, modes2=48, width=32, in_channels=1).cuda()

# 创建模拟输入数据
input_tensor = torch.randn(8, 1, 112, 192)  # [batch_size, channel, height, width]
input_tensor = input_tensor.cuda()

# 执行前向传播
output = model(input_tensor)

# 检查输出
print(output.shape)