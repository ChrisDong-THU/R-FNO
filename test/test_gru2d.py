import torch
import torch.nn as nn

import sys
import os

from omegaconf import OmegaConf

filename = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(filename)

from model import RFNO, timer

cfg = OmegaConf.load(f'D:/paper_projects/R-FNOv2/conf/rfno-1.yaml')
model = RFNO(fno=cfg.fno, gru=cfg.gru, sync=True).cuda()

# 创建模拟输入数据
input_tensor = torch.randn(8, 3, 1, 112, 192) # [batch_size, input_size, c, h, w]

input_tensor = input_tensor.cuda()

# 执行前向传播
timer.set_enabled(True)
output1 = model(input_tensor)

for k, v in timer.get_timing_stat().items():
    print('Function "%s" takes %.1fms' % (k, v))

# 检查输出
# print(output1.shape)
# print(output2.shape)