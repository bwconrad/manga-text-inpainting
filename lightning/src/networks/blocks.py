import torch
import torch.nn as nn

def spectral_norm(x, apply=True):
    return nn.utils.spectral_norm(x) if apply else x

class ResnetBlockIns(nn.Module):
    '''
    From: https://github.com/knazeri/edge-connect/blob/master/src/networks.py
    '''
    def __init__(self, dim, dilation=1, use_spectral_norm=False):
        super(ResnetBlockIns, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(dilation),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, 
                                    dilation=dilation, bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(dim, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, 
                                    dilation=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(dim, track_running_stats=False),
        )

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class ResnetBlockBatch(nn.Module):
    '''
    From: https://github.com/knazeri/edge-connect/blob/master/src/networks.py
    '''
    def __init__(self, dim, dilation=1, use_spectral_norm=False):
        super(ResnetBlockBatch, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(dilation),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, 
                                    dilation=dilation, bias=not use_spectral_norm), use_spectral_norm),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),

            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, 
                                    dilation=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.BatchNorm2d(dim),
        )

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class PreActResnetBlockBatch(nn.Module):
    '''
    From: https://github.com/knazeri/edge-connect/blob/master/src/networks.py
    '''
    def __init__(self, dim, dilation=1, use_spectral_norm=False):
        super(PreActResnetBlockBatch, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(dilation),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, 
                                    dilation=dilation, bias=not use_spectral_norm), use_spectral_norm),

            nn.ReflectionPad2d(1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, 
                                    dilation=1, bias=not use_spectral_norm), use_spectral_norm),
        )

    def forward(self, x):
        out = x + self.conv_block(x)
        return out