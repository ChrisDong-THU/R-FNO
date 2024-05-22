import torch
import torch.nn as nn


class GRU2d(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=128):
        '''GRU2d

        :param int hidden_dim: 隐状态通道数, defaults to 128
        :param int input_dim: 输入通道数, defaults to 128
        '''
        super().__init__()
        
        self.hidden_dim, self.input_dim = hidden_dim, input_dim

        self.convz1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1, 5), padding=(0, 2)) # 保证特征图大小不变
        self.convr1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convq1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1, 5), padding=(0, 2))

        self.convz2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5, 1), padding=(2, 0))
        self.convr2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5, 1), padding=(2, 0))
        self.convq2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5, 1), padding=(2, 0))

    def forward(self, x, h):
        # horizontal
        hx = torch.cat([h, x], dim=1) # 通道维度拼接
        z = torch.sigmoid(self.convz1(hx)) # 更新门
        r = torch.sigmoid(self.convr1(hx)) # 重置门
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1)))
        h = (1-z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)))
        h = (1-z) * h + z * q

        h = torch.nan_to_num(h)
        return h