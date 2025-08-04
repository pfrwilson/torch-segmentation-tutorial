import torch


# build a model 
from torch import nn 


class EncoderBlock(nn.Module): 
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1): 
        super().__init__()
        self.norm = nn.BatchNorm3d(in_channels)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.activation = nn.ReLU()
        self.downsample = nn.MaxPool3d(2)

    def forward(self, x): 
        x = self.norm(x)
        x = self.conv(x)
        x = self.activation(x)
        x = self.downsample(x)
        return x


class DecoderBlock(nn.Module): 
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1): 
        super().__init__()
        self.norm = nn.BatchNorm3d(in_channels)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.activation = nn.ReLU()

    def upsample(self, x): 
        B, N, H, W, D = x.shape
        return torch.nn.functional.interpolate(x, (H * 2, W * 2, D * 2))

    def forward(self, x):
        x = self.norm(x)
        x = self.conv(x)
        x = self.activation(x)
        x = self.upsample(x)
        return x


class EncoderDecoderNetwork(nn.Module): 
    def __init__(self): 
        super().__init__()
        
        self.layers = nn.Sequential(
            EncoderBlock(1, 16), 
            EncoderBlock(16, 32), 
            EncoderBlock(32, 64),  
            DecoderBlock(64, 32), 
            DecoderBlock(32, 16), 
            DecoderBlock(16, 2)
        )

    def forward(self, x): 
        return self.layers(x)