import torch
import torch.nn as nn
import torch.nn.functional as F

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
                                    dilation=dilation, bias=False), use_spectral_norm),
            nn.InstanceNorm2d(dim, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, 
                                    dilation=1, bias=False), use_spectral_norm),
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
                                    dilation=dilation, bias=False), use_spectral_norm),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),

            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, 
                                    dilation=1, bias=False), use_spectral_norm),
            nn.BatchNorm2d(dim),
        )

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class PreActResnetBlockBatch(nn.Module):
    def __init__(self, dim, dilation=1, use_spectral_norm=False):
        super(PreActResnetBlockBatch, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(dilation),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, 
                                    dilation=dilation, bias=False), use_spectral_norm),

            nn.ReflectionPad2d(1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, 
                                    dilation=1, bias=False), use_spectral_norm),
        )

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class ConvBlock(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dilation=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=dilation, dilation=dilation)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(True)

    def forward(self,x):
        out = self.relu(self.bn(self.conv(x)))
        return out

def upsample_like(inp,target):
    ''' From https://github.com/NathanUA/U-2-Net/blob/master/model/u2net.py '''
    inp = F.interpolate(inp, size=target.shape[2:], mode='bilinear', align_corners=False)
    return inp

class ResUBlock(nn.Module):
    def __init__(self, depth, in_ch=3, mid_ch=12, out_ch=3):
        super(ResUBlock, self).__init__()

        # Encoder
        self.encoder = nn.ModuleList([
            ConvBlock(in_ch, out_ch),
            ConvBlock(out_ch, mid_ch),   
        ])
        for d in range(depth-2):
            self.encoder.append(nn.Sequential(
                nn.MaxPool2d(2, stride=2, ceil_mode=True),
                ConvBlock(mid_ch, mid_ch)
            ))
        self.encoder.append(
            ConvBlock(mid_ch, mid_ch, dilation=2)
        )

        # Decoder
        self.decoder = nn.ModuleList([])
        for d in range(depth-2):
            self.decoder.append(ConvBlock(mid_ch*2, mid_ch))
        self.decoder.append(
            ConvBlock(mid_ch*2, out_ch)
        )

    def forward(self, x):
        # Encoder
        x_encoder = [self.encoder[0](x)]
        for e_layer in self.encoder[1:]: 
            x_encoder.append(
                e_layer(x_encoder[-1])
            )
        x_encoder.reverse() # Reverse for simpler indexing

        # Decoder
        out = x_encoder[0] # Previous layer output
        for i, d_layer in enumerate(self.decoder[:-1]):
            skip = x_encoder[i+1] # Skip connection w/ encoder
            out = d_layer(torch.cat((out, skip), 1))
            out = upsample_like(out, x_encoder[i+2])
        out = self.decoder[-1](torch.cat((out, x_encoder[-2]), 1))

        return out + x_encoder[-1] 

class ResUBlockFlat(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(ResUBlockFlat, self).__init__()

        self.encoder = nn.ModuleList([
            ConvBlock(in_ch, out_ch, dilation=1),
            ConvBlock(out_ch, mid_ch, dilation=1),
            ConvBlock(mid_ch, mid_ch, dilation=2),
            ConvBlock(mid_ch, mid_ch, dilation=4),
            ConvBlock(mid_ch, mid_ch, dilation=8),
        ])

        self.decoder = nn.ModuleList([
            ConvBlock(mid_ch*2, mid_ch, dilation=4),
            ConvBlock(mid_ch*2, mid_ch, dilation=2),
            ConvBlock(mid_ch*2, out_ch, dilation=1)
        ])


    def forward(self, x):
        # Encoder
        x0 = self.encoder[0](x)
        x1 = self.encoder[1](x0)
        x2 = self.encoder[2](x1)
        x3 = self.encoder[3](x2)
        x4 = self.encoder[4](x3)

        # Decoder
        out3 = self.decoder[0](torch.cat((x4, x3), 1))
        out2 = self.decoder[1](torch.cat((out3, x2), 1))
        out1 = self.decoder[2](torch.cat((out2, x1), 1))

        return out1 + x0
        
        
        






