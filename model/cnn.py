# cf. Fukami Vonoroi-CNN
import torch
import torch.nn as nn
import torch.nn.functional as F

class RecFieldCNN(nn.Module):
    def __init__(self, in_channels=2, out_channels=1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=48, kernel_size=7, padding='same'),
            nn.GELU(),
            nn.Conv2d(in_channels=48, out_channels=48, kernel_size=7, padding='same'),
            nn.GELU(),
            nn.Conv2d(in_channels=48, out_channels=48, kernel_size=7, padding='same'),
            nn.GELU(),
            nn.Conv2d(in_channels=48, out_channels=48, kernel_size=7, padding='same'),
            nn.GELU(),
            nn.Conv2d(in_channels=48, out_channels=48, kernel_size=7, padding='same'),
            nn.GELU(),
            nn.Conv2d(in_channels=48, out_channels=48, kernel_size=7, padding='same'),
            nn.GELU(),
            nn.Conv2d(in_channels=48, out_channels=48, kernel_size=7, padding='same'),
            nn.GELU(),
            nn.Conv2d(in_channels=48, out_channels=out_channels, kernel_size=3, padding='same')
        )
        
    def forward(self, x):
        return self.net(x)