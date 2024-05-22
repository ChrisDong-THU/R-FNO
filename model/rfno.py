import torch
import torch.nn as nn

from .fno import FNO2d
from .gru import GRU2d
from .utils import timer

import time
    
class RFNO(nn.Module):
    def __init__(self, fno, gru, in_channels=1, out_channels=1, img_size=(112, 192), sync=False, model_path=None) -> None:
        '''RFNO2d

        :param _type_ fno: fno参数
        :param _type_ gru: gru参数
        :param int input_size: 序列数量, defaults to 3
        :param int in_channels: 输入通道数, defaults to 1
        :param int out_channels: 输出通道数, defaults to 1
        :param bool sync: 是否同步模式, defaults to False
        :param str model_path: 预训练模型路径, defaults to None
        '''
        super().__init__()
        self.input_size = gru.input_size
        self.img_size = img_size
        self.sync = sync
        self.gru_cfg = gru
        
        # 异步模式下为预训练好的FNO2d模型
        self.fno2d = FNO2d(modes1=fno.modes1, modes2=fno.modes2, width=fno.width, in_channels=in_channels, out_channels=out_channels)
        if not self.sync:
            self.fno2d.load_state_dict(torch.load(model_path)['state_dict'])

        self.gru2d = GRU2d(hidden_dim=gru.hidden_dim, input_dim=gru.input_dim)
        self.fc = nn.Sequential(
            nn.Linear(gru.hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, out_channels)
        )
        
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(batch_size, self.gru_cfg.hidden_dim, *self.img_size).zero_().cuda()

        return hidden
    
    def decoder(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.fc(x)
        x = x.permute(0, 3, 1, 2)
        
        return x
    
    @timer.timer_func
    def forward(self, x):
        # [batch_size, input_size, c, h, w]
        # 先对整个批次进行fno编码
        x_encoded = self.fno2d.encoder(x.view(-1, *x.shape[2:])) # [batch_size*input_size, c, h, w]
        x_encoded = x_encoded.view(*x.shape[:2], *x_encoded.shape[1:]) # [batch_size, input_size, c, h, w]
        
        if not self.sync: x_encoded = x_encoded.detach()
        # 初始化隐藏状态
        h = self.init_hidden(batch_size=x.shape[0]) # [batch_size, hidden_dim, h, w]
        # 循环处理输入序列
        for i in range(self.input_size):
            h = self.gru2d(x_encoded[:,i], h)
        
        # 解码
        x_decoded = self.decoder(h)
        
        return x_decoded