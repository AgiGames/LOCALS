import torch
import torch.nn as nn

from .blocks import ResnetBlock, ConvBlock

class LOCALSn(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.model = nn.Sequential(
            ConvBlock(input_channels=3, output_channels=16, kernel_size=2, stride=2), # 224
            ResnetBlock(input_channels=16, output_channels=32), # 112
            ResnetBlock(input_channels=32, output_channels=64), # 56
            ConvBlock(input_channels=64, output_channels=128, kernel_size=2, stride=2), # 28
            ResnetBlock(input_channels=128, output_channels=256), # 14
            ResnetBlock(input_channels=256, output_channels=256), # 7
        )
        
        # 1x1 conv is important here because we do not want to do feature extraction in heads
        
        self.head = nn.Sequential(
            ConvBlock(input_channels=256, output_channels=3, activation=False),
        )
        
    def forward(self, x):
        out = self.model(x)
        out = self.head(out)
        out = out.permute(0, 2, 3, 1) # [B, 7, 7, 3]
        out = torch.sigmoid(out)
        return out
    
class LOCALSs(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.model = nn.Sequential(
            ConvBlock(input_channels=3, output_channels=16, kernel_size=3, padding=1), # 448
            ResnetBlock(input_channels=16, output_channels=32), # 224
            ResnetBlock(input_channels=32, output_channels=64), # 112
            ResnetBlock(input_channels=64, output_channels=128), # 56
            ResnetBlock(input_channels=128, output_channels=256), # 28
            ResnetBlock(input_channels=256, output_channels=512), # 14
            ResnetBlock(input_channels=512, output_channels=256), # 7
        )
        
        # 1x1 conv is important here because we do not want to do feature extraction in heads
        
        self.loc_head = nn.Sequential(
            ConvBlock(input_channels=256, output_channels=64),
            ConvBlock(input_channels=64, output_channels=16),
            ConvBlock(input_channels=16, output_channels=2, activation=False),
        )
        
        self.class_head = nn.Sequential(
            ConvBlock(input_channels=256, output_channels=1, activation=False),
        )
        
    def forward(self, x):
        out =  self.model(x)
        loc_info = self.loc_head(out) # [B, 2, 7, 7]
        class_info = self.class_head(out) # [B, 1, 7, 7]
        out = torch.concatenate([loc_info, class_info], dim=1) # [B, 3, 7, 7]
        out = out.permute(0, 2, 3, 1) # [B, 7, 7, 3]
        out = torch.sigmoid(out)
        return out