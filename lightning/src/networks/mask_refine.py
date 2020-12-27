import torch
import torch.nn as nn

from .blocks import *

class MREncoderDecoder(nn.Module):
    def __init__(self, nf=32, residual_blocks=4, dilation=2, use_spectral_norm=True):
        super(MREncoderDecoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            spectral_norm(nn.Conv2d(in_channels=2, out_channels=nf, kernel_size=7, padding=0), use_spectral_norm),
            nn.BatchNorm2d(nf),
            nn.ReLU(True),

            spectral_norm(nn.Conv2d(in_channels=nf, out_channels=nf*2, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.BatchNorm2d(nf*2),
            nn.ReLU(True),

            spectral_norm(nn.Conv2d(in_channels=nf*2, out_channels=nf*4, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.InstanceNorm2d(256, track_running_stats=False),
            nn.ReLU(True)
        )

        blocks = []
        for _ in range(residual_blocks):
            block = ResnetBlockBatch(nf*4, dilation=dilation, use_spectral_norm=use_spectral_norm)
            blocks.append(block)

        self.middle = nn.Sequential(*blocks)

        self.decoder = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(in_channels=nf*4, out_channels=nf*2, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.BatchNorm2d(nf*2),
            nn.ReLU(True),

            spectral_norm(nn.ConvTranspose2d(in_channels=nf*2, out_channels=nf, kernel_size=4, stride=2, padding=1), use_spectral_norm),
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
