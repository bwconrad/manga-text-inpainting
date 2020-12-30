import torch
import torch.nn as nn

from .blocks import *

class MREncoderDecoder(nn.Module):
    def __init__(self, nf=32, residual_blocks=4, dilation=2):
        super(MREncoderDecoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=2, out_channels=nf, kernel_size=7, padding=0, bias=False),
            nn.BatchNorm2d(nf),
            nn.ReLU(True),

            nn.Conv2d(in_channels=nf, out_channels=nf*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(nf*2),
            nn.ReLU(True),

            nn.Conv2d(in_channels=nf*2, out_channels=nf*4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(nf*4, track_running_stats=False),
            nn.ReLU(True)
        )

        blocks = []
        for _ in range(residual_blocks):
            block = PreActResnetBlockBatch(nf*4, dilation=dilation, use_spectral_norm=False)
            blocks.append(block)

        self.middle = nn.Sequential(*blocks)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=nf*4, out_channels=nf*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(nf*2),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=nf*2, out_channels=nf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(nf),
            nn.ReLU(True),

            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=nf, out_channels=1, kernel_size=7, padding=0),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        x = torch.sigmoid(x)
        return x


class U2Net(nn.Module):
    '''
    Small varient
    https://arxiv.org/pdf/2005.09007.pdf
    '''
    def __init__(self, nf=64):
        super(U2Net, self).__init__()
        in_ch = 2
        out_ch = 1

        # Encoder
        self.e_block1 = ResUBlock(7, in_ch, 16, nf)
        self.e_block2 = nn.Sequential(
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
            ResUBlock(6, nf, 16, nf)
        )
        self.e_block3 = nn.Sequential(
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
            ResUBlock(5, nf, 16, nf)
        )
        self.e_block4 = nn.Sequential(
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
            ResUBlock(4, nf, 16, nf)
        )
        self.e_block5 = nn.Sequential(
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
            ResUBlockFlat(nf, 16, nf)
        )
        self.e_block6 = nn.Sequential(
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
            ResUBlockFlat(nf, 16, nf)
        )

        # Decoder
        self.d_block5 = ResUBlockFlat(2*nf, 16, nf)
        self.d_block4 = ResUBlock(4, 2*nf, 16, nf)
        self.d_block3 = ResUBlock(5, 2*nf, 16, nf)
        self.d_block2 = ResUBlock(6, 2*nf, 16, nf)
        self.d_block1 = ResUBlock(7, 2*nf, 16, nf)

        # Outputs
        self.side1 = nn.Conv2d(nf, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(nf, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(nf, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(nf, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(nf, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(nf, out_ch, 3, padding=1)
        self.out = nn.Conv2d(6, out_ch, 1)

    def forward(self, x):
        # Encoder
        e_1 = self.e_block1(x)
        e_2 = self.e_block2(e_1)
        e_3 = self.e_block3(e_2)
        e_4 = self.e_block4(e_3)
        e_5 = self.e_block5(e_4)
        e_6 = self.e_block6(e_5)
        e_6 = upsample_like(e_6, e_5)
        
        # Decoder
        d_5 = self.d_block5(torch.cat((e_6, e_5), 1))
        d_5 = upsample_like(d_5, e_4)
        d_4 = self.d_block4(torch.cat((d_5, e_4), 1))
        d_4 = upsample_like(d_4, e_3)
        d_3 = self.d_block3(torch.cat((d_4, e_3), 1))
        d_3 = upsample_like(d_3, e_2)
        d_2 = self.d_block2(torch.cat((d_3, e_2), 1))
        d_2 = upsample_like(d_2, e_1)
        d_1 = self.d_block1(torch.cat((d_2, e_1), 1))

        # Outputs at each scale
        out1 = self.side1(d_1)
        out2 = upsample_like(self.side2(d_2), d_1)
        out3 = upsample_like(self.side3(d_3), d_1)
        out4 = upsample_like(self.side4(d_4), d_1)
        out5 = upsample_like(self.side5(d_5), d_1)
        out6 = upsample_like(self.side6(e_6), d_1)

        # Combine all scales into final output
        out = self.out(torch.cat((out1, out2, out3, out4, out5, out6), 1))
        out = torch.sigmoid(out)

        return out

