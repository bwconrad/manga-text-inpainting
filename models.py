import torch
import torch.nn as nn
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
            block += [padding(1)]

        block += [spectral_norm(nn.Conv2d(dim, dim, kernel_size=3, padding=p, dilation=dilation, bias=not use_spectral_norm), use_spectral_norm),
                  norm_layer(dim),
                  activation]
        
        if use_dropout:
            block += [nn.Dropout(0.5)]

        if padding:
            block += [padding(1)]

        block += [spectral_norm(nn.Conv2d(dim, dim, kernel_size=3, padding=p, dilation=dilation, bias=not use_spectral_norm), use_spectral_norm),
                  norm_layer(dim)]

        return nn.Sequential(*block)

    def forward(self, x):
        out = x + self.block(x)
        return out

class GlobalGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d, 
                 padding_type='reflect', use_dropout=False, use_spectral_norm=False, dilation=1):
        assert(n_blocks >= 0)
        super(GlobalGenerator, self).__init__()        
        activation = nn.ReLU(True)        

        model = [nn.ReflectionPad2d(3), 
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), 
                 norm_layer(ngf), 
                 activation]

        # Encoder
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
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
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                       norm_layer(int(ngf * mult / 2)), 
                       activation]

        model += [nn.ReflectionPad2d(3), 
                  nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), 
                  nn.Tanh()] 

        self.model = nn.Sequential(*model)
            
    def forward(self, input):
        return self.model(input) 

class UNetSkipBlock(nn.Module):
    def __init__(self, outer_ch, inner_ch, input_ch=None,
                 submodule=None, outermost=False, innermost=False,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UNetSkipBlock, self).__init__()
        self.outermost = outermost

        # No bias since BatchNorm2d has affine parameters
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        if input_ch is None:
            input_ch = outer_ch

        down_conv = nn.Conv2d(input_ch, inner_ch, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        down_relu = nn.LeakyReLU(0.2, True)
        down_norm = norm_layer(inner_ch)

        up_relu = nn.ReLU(True)
        up_norm = norm_layer(outer_ch)

        # First and last block
        if outermost:
            up_conv = nn.ConvTranspose2d(inner_ch * 2, outer_ch,
                                         kernel_size=4, stride=2, 
                                         padding=1)
            down = [down_conv]
            up = [up_relu, up_conv, nn.Tanh()]
            model = down + [submodule] + up

        # Middle bottleneck
        elif innermost:
            up_conv = nn.ConvTranspose2d(inner_ch, outer_ch,
                                         kernel_size=4, stride=2, 
                                         padding=1, bias=use_bias)
            down = [down_relu, down_conv]
            up = [up_relu, up_conv, up_norm]
            model = down + up 

        else:
            up_conv = nn.ConvTranspose2d(inner_ch * 2, outer_ch,
                                         kernel_size=4, stride=2,
                                         padding=1, bias=use_bias)
            down = [down_relu, down_conv, down_norm]
            up = [up_relu, up_conv, up_norm] 
            model = down + [submodule] + up
            
            if use_dropout:
                model += [nn.Dropout(0.5)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)

class UNetGenerator(nn.Module):
    def __init__(self, input_ch, output_ch, n_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UNetGenerator, self).__init__()

        # Build from innermost to outermost
        unet_block = UNetSkipBlock(ngf * 8, ngf * 8, input_ch=None, submodule=None,
                                   norm_layer=norm_layer, innermost=True)
        for i in range(n_downs-5):
            unet_block = UNetSkipBlock(ngf * 8, ngf * 8, input_ch=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UNetSkipBlock(ngf * 4, ngf * 8, input_ch=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UNetSkipBlock(ngf * 2, ngf * 4, input_ch=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UNetSkipBlock(ngf, ngf * 2, input_ch=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UNetSkipBlock(output_ch, ngf, input_ch=input_ch, submodule=unet_block, outermost=True, norm_layer=norm_layer)

    def forward(self, inp):
        return self.model(inp)    
                               
class PatchDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        super(PatchDiscriminator, self).__init__()

        # No bias since BatchNorm2d has affine parameters
        if type(norm_layer) == functools.partial: 
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1),
                 nn.LeakyReLU(0.2, True)]

        nf_mult = 1
        nf_mult_prev = 1
        for i in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**i, 8)
            model += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=2, padding=1, bias=use_bias),
                      norm_layer(ndf * nf_mult),
                      nn.LeakyReLU(0.2, True)]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        model += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=1, padding=1, bias=use_bias),
                      norm_layer(ndf * nf_mult),
                      nn.LeakyReLU(0.2, True)]

        model += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)]
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


## TODO
class MultiScaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, num_D=3, norm_layer=nn.BatchNorm2d):
        super(MultiScaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers

        for i in range(num_D):
            netD = PatchDiscriminator(input_nc=input_nc, ndf=ndf, n_layers=n_layers, norm_layer=norm_layer)
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

        return np.sum(result)

