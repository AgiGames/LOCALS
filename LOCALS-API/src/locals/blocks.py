import torch.nn as nn

'''
This block models the skip connection introduced in the Resnet paper 2015.
'''

class ResnetBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=output_channels, out_channels=output_channels, kernel_size=3, stride=2, padding=1)
        self.downsamp_conv = nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.bn2 = nn.BatchNorm2d(output_channels)
        self.downsamp_bn = nn.BatchNorm2d(output_channels)
        
    def forward(self, x):
        # feature extraction
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        # reduce H and W by 2
        out = self.conv2(out)
        out = self.bn2(out)
        
        # bring x to required number of channels
        downsamp_x = self.downsamp_conv(x)
        downsamp_x = self.downsamp_bn(downsamp_x)
        
        # skip connection
        return self.relu(out + downsamp_x)
    
'''
Simple 1x1 convolution block, nothing much to see here.
'''

class ConvBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=1, padding=0, stride=1, activation=True):
        super().__init__()
        
        self.channel_downsamp_conv = nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        self.channel_downsamp_bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()
        self.activation = activation
        
    def forward(self, x):
        out = self.channel_downsamp_conv(x)
        out = self.channel_downsamp_bn(out)
        if self.activation:
            out = self.relu(out)
        
        return out