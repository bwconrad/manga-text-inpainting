import torch
import torch.nn as nn
import torchvision.models as models
import functools
import numpy as np

def spectral_norm(module, use=True):
    if use:
        return nn.utils.spectral_norm(module)

    return module

class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False, use_spectral_norm=False, dilation=1):
        super(ResnetBlock, self).__init__()
        self.block = self.build_block(dim, padding_type, norm_layer, activation, use_dropout, use_spectral_norm, dilation)

    def build_block(self, dim, padding_type, norm_layer, activation, use_dropout, use_spectral_norm, dilation):
        block = []

        # Choose padding type
        if padding_type == 'reflect':
            padding = nn.ReflectionPad2d
            p = 0
        elif padding_type == 'replicate':
            padding = nn.ReplicationPad2d
            p = 0
        elif padding_type == 'zero':
            padding = None
            p = 1
        else:
            raise NotImplementedError('padding [{}] is not implemented'.format(padding_type))

        # Create block
        if padding:
            block += [padding(dilation)]

        block += [spectral_norm(nn.Conv2d(dim, dim, kernel_size=3, padding=p, dilation=dilation, bias=not use_spectral_norm), use_spectral_norm),
                  norm_layer(dim),
                  activation]
        
        if use_dropout:
            block += [nn.Dropout(0.5)]

        if padding:
            block += [padding(1)]

        block += [spectral_norm(nn.Conv2d(dim, dim, kernel_size=3, padding=p, dilation=1, bias=not use_spectral_norm), use_spectral_norm),
                  norm_layer(dim)]

        return nn.Sequential(*block)

    def forward(self, x):
        out = x + self.block(x)
        return out

class GlobalGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d, 
                 padding_type='reflect', use_dropout=False, use_spectral_norm=False, dilation=1, kernel_size=3):
        assert(n_blocks >= 0)
        super(GlobalGenerator, self).__init__()        
        activation = nn.ReLU(True)        

        model = [nn.ReflectionPad2d(3), 
                 spectral_norm(nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), use_spectral_norm), 
                 norm_layer(ngf), 
                 activation]

        # Encoder
        for i in range(n_downsampling):
            mult = 2**i
            model += [spectral_norm(nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=kernel_size, stride=2, padding=1), use_spectral_norm),
                      norm_layer(ngf * mult * 2), 
                      activation]

        # Resblocks
        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer,
                                  use_dropout=use_dropout, use_spectral_norm = use_spectral_norm, dilation=dilation)]
        
        # Decoder        
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [spectral_norm(nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=kernel_size, stride=2, padding=1, 
                      output_padding=1 if kernel_size==3 else 0), use_spectral_norm),
                       norm_layer(int(ngf * mult / 2)), 
                       activation]

        model += [nn.ReflectionPad2d(3), 
                  nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), 
                  nn.Tanh()] 

        self.model = nn.Sequential(*model)
            
    def forward(self, input):
        return self.model(input) 
                               
class PatchDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_spectral_norm=False):
        super(PatchDiscriminator, self).__init__()

        model = [spectral_norm(nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1, bias= not use_spectral_norm), use_spectral_norm),
                 nn.LeakyReLU(0.2, True)]

        nf_mult = 1
        nf_mult_prev = 1
        for i in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**i, 8)
            model += [spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
                      norm_layer(ndf * nf_mult),
                      nn.LeakyReLU(0.2, True)]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        model += [spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
                      norm_layer(ndf * nf_mult),
                      nn.LeakyReLU(0.2, True)]

        model += [spectral_norm(nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm)]
        self.model = nn.Sequential(*model)

    def forward(self, inp):
        return self.model(inp)

class PixelDiscriminator(nn.Module):
    '''
    1x1 PatchGAN discriminator
    '''
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        super(PixelDiscriminator, self).__init__()

        # No bias since BatchNorm2d has affine parameters
        if type(norm_layer) == functools.partial: 
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
                 nn.LeakyReLU(0.2, True),
                 nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
                 norm_layer(ndf * 2),
                 nn.LeakyReLU(0.2, True),
                 nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0)]
        
        self.model = nn.Sequential(*model)


    def forward(self, inp):
        return self.model(inp)

class MultiScaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, num_D=3, norm_layer=nn.BatchNorm2d, use_spectral_norm=False):
        super(MultiScaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers

        for i in range(num_D):
            netD = PatchDiscriminator(input_nc=input_nc, ndf=ndf, n_layers=n_layers, norm_layer=norm_layer, use_spectral_norm=use_spectral_norm)
            setattr(self, 'layer'+str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)

    def single_forward(self, model, inp):
        return [model(inp)]

    def forward(self, inp):
        result = []
        input_downsampled = inp

        for i in range(self.num_D):
            model = getattr(self, 'layer'+str(self.num_D-1-i))
            result.append(self.single_forward(model, input_downsampled))

            if i != (self.num_D-1):
                input_downsampled = self.downsample(input_downsampled)

        return result

class VGG19(torch.nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        features = models.vgg19(pretrained=True).features
        self.relu1_1 = torch.nn.Sequential()
        self.relu1_2 = torch.nn.Sequential()

        self.relu2_1 = torch.nn.Sequential()
        self.relu2_2 = torch.nn.Sequential()

        self.relu3_1 = torch.nn.Sequential()
        self.relu3_2 = torch.nn.Sequential()
        self.relu3_3 = torch.nn.Sequential()
        self.relu3_4 = torch.nn.Sequential()

        self.relu4_1 = torch.nn.Sequential()
        self.relu4_2 = torch.nn.Sequential()
        self.relu4_3 = torch.nn.Sequential()
        self.relu4_4 = torch.nn.Sequential()

        self.relu5_1 = torch.nn.Sequential()
        self.relu5_2 = torch.nn.Sequential()
        self.relu5_3 = torch.nn.Sequential()
        self.relu5_4 = torch.nn.Sequential()

        for x in range(2):
            self.relu1_1.add_module(str(x), features[x])

        for x in range(2, 4):
            self.relu1_2.add_module(str(x), features[x])

        for x in range(4, 7):
            self.relu2_1.add_module(str(x), features[x])

        for x in range(7, 9):
            self.relu2_2.add_module(str(x), features[x])

        for x in range(9, 12):
            self.relu3_1.add_module(str(x), features[x])

        for x in range(12, 14):
            self.relu3_2.add_module(str(x), features[x])

        for x in range(14, 16):
            self.relu3_3.add_module(str(x), features[x])

        for x in range(16, 18):
            self.relu3_4.add_module(str(x), features[x])

        for x in range(18, 21):
            self.relu4_1.add_module(str(x), features[x])

        for x in range(21, 23):
            self.relu4_2.add_module(str(x), features[x])

        for x in range(23, 25):
            self.relu4_3.add_module(str(x), features[x])

        for x in range(25, 27):
            self.relu4_4.add_module(str(x), features[x])

        for x in range(27, 30):
            self.relu5_1.add_module(str(x), features[x])

        for x in range(30, 32):
            self.relu5_2.add_module(str(x), features[x])

        for x in range(32, 34):
            self.relu5_3.add_module(str(x), features[x])

        for x in range(34, 36):
            self.relu5_4.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = x.repeat(1, 3, 1, 1)

        relu1_1 = self.relu1_1(x)
        relu1_2 = self.relu1_2(relu1_1)

        relu2_1 = self.relu2_1(relu1_2)
        relu2_2 = self.relu2_2(relu2_1)

        relu3_1 = self.relu3_1(relu2_2)
        relu3_2 = self.relu3_2(relu3_1)
        relu3_3 = self.relu3_3(relu3_2)
        relu3_4 = self.relu3_4(relu3_3)

        relu4_1 = self.relu4_1(relu3_4)
        relu4_2 = self.relu4_2(relu4_1)
        relu4_3 = self.relu4_3(relu4_2)
        relu4_4 = self.relu4_4(relu4_3)

        relu5_1 = self.relu5_1(relu4_4)
        relu5_2 = self.relu5_2(relu5_1)
        relu5_3 = self.relu5_3(relu5_2)
        relu5_4 = self.relu5_4(relu5_3)

        out = {
            'relu1_1': relu1_1,
            'relu1_2': relu1_2,

            'relu2_1': relu2_1,
            'relu2_2': relu2_2,

            'relu3_1': relu3_1,
            'relu3_2': relu3_2,
            'relu3_3': relu3_3,
            'relu3_4': relu3_4,

            'relu4_1': relu4_1,
            'relu4_2': relu4_2,
            'relu4_3': relu4_3,
            'relu4_4': relu4_4,

            'relu5_1': relu5_1,
            'relu5_2': relu5_2,
            'relu5_3': relu5_3,
            'relu5_4': relu5_4,
        }
        return out
