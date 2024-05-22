# cf. Physics-Informed Neural Operator for Learning Partial Differential Equations
import torch
import torch.nn as nn
import torch.nn.functional as F

# 频谱卷积
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        '''2D Fourier layer. It does FFT, linear transform, and Inverse FFT.

        :param int in_channels: 输入通道数
        :param int out_channels: 输出通道数
        :param int modes1: 傅里叶模态数1
        :param int modes2: 傅里叶模态数2
        '''
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 2D时两个傅里叶模态数量，由Nyquist定理，最后一维最大floor(N/2) + 1
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # 复数乘法
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # 实数傅里叶变换，只包含非负频率部分的信息，最后一维最大floor(N/2) + 1
        x_ft = torch.fft.rfft2(x) # NCHW->NCH(W/2+1)
        
        # 第二维傅里叶模态进行截断，第一维傅里叶模态对称性计算
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat,
                                 device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # 反傅里叶变换回物理空间，大小(h,w)
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        
        return x


class FNO2d(nn.Module):
    def __init__(self, modes1=32, modes2=40, width=32, in_channels=1, out_channels=1):
        '''2D傅里叶神经算子
        1. self.fc0提升通道维度
        2. 4层积分算子，self.w物理空间卷积，self.conv频域卷积，u' = (W + R)(u)
        3. self.fc1和self.fc2投射到输出空间

        :param int modes1: 傅里叶模态数1
        :param int modes2: 傅里叶模态数2
        :param int width: 傅里叶层通道数
        :param int in_channels: FNO2d输入通道数, defaults to 3, Vonoroi嵌入时多一个掩膜嵌入通道in_channels=4
        :param int out_channels: FNO2d输出图像NCHW的通道数, defaults to 1
        '''
        super().__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.fc0 = nn.Linear(in_channels, self.width)
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def encoder(self, x):
        # ->NHWC，全连接层的输入张量需要最内层为特征维度
        x = self.fc0(x.permute(0, 2, 3, 1))
        # ->NCHW，卷积层常用顺序
        x = x.permute(0, 3, 1, 2)

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2
        
        return x
    
    def decoder(self, x):
        # ->NHWC
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        # ->NCHW
        x = x.permute(0, 3, 1, 2)
        
        return x
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        
        return x