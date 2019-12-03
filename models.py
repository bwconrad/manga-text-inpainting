import torch
import torch.nn as nn
import functools

class InputBlock(nn.Module):
	'''
	First encoder block
	'''
	def __init__(self, in_ch, out_ch, norm_layer, use_bias):
		super(InputBlock, self).__init__()
		self.block = nn.Sequential(
			nn.ReflectionPad2d(3),
			nn.Conv2d(in_ch, out_ch, kernel_size=7, padding=0, bias=use_bias),
			norm_layer(out_ch),
			nn.ReLU(True)
		)

	def forward(self, x):
		x = self.block(x)
		return out

class DownBlock(nn.Module):
	'''
	Downsampling block
	'''
	def __init__(self, in_ch, out_ch, norm_layer, use_bias):
		super(DownBlock, self).__init__()
		self.block = nn.Sequential(
			nn.Conv2d(in_ch, out_ch, kernel_size=3,
					  stride=2, padding=1, bias=use_bias),
			norm_layer(out_ch),
			nn.ReLU(True)
		)

	def forward(self, x):
		x = self.block(x)
		return x

class UpBlock(nn.Module):
	'''
	Upsampling block
	'''
	def __init__(self, in_ch, out_ch, norm_layer, use_bias):
		super(UpBlock, self).__init__()
		self.block = nn.Sequential(
			nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3,
					  		   stride=2, padding=1, output_padding=1, 
					  		   bias=use_bias),
			norm_layer(out_ch),
			nn.ReLU(True)
		)

	def forward(self, x):
		x = self.block(x)
		return x

class OutBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_ch, out_ch, kernel_size=7, padding=0),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.block(x)
        return x

class ResBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResBlock, self).__init__()
        self.block = self.build_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
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

        block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                  norm_layer(dim),
                  nn.ReLU(True)]
        
        if use_dropout:
            block += [nn.Dropout(0.5)]

        if padding:
            block += [padding(1)]

        block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                  norm_layer(dim)]

        return nn.Sequential(*block)

    def forward(self, x):
        out = x + self.block(x)
        out = nn.ReLU(True)(out)
        return out

class ResNetGenerator(nn.Module):
    def __init__(self, in_ch, out_ch, ngf=64, norm_layer=nn.BatchNorm2d, 
                 use_dropout=False, n_blocks=9, padding_type='reflect'):
        super(Generator, self).__init__()

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.ngf = ngf

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.input_block = InputBlock(in_ch, ngf, norm_layer, use_bias)
        self.down1 = DownBlock(ngf, ngf*2, norm_layer, use_bias)
        self.down2 = DownBlock(ngf*2, ngf*4, norm_layer, use_bias)

        model = []
        for i in range(n_blocks):
            model += [ResBlock(ngf*4, padding_type=padding_type, norm_layer=norm_layer,
                      use_dropout=use_dropout, use_bias=use_bias)]
        self.resblocks = nn.Sequential(*model)

        self.up1 = UpBlock(ngf*4, ngf*2, norm_layer, use_bias)
        self.up2 = UpBlock(ngf*2, ngf, norm_layer, use_bias)
        self.output_block = OutBlock(ngf. out_ch)


    def forward(self, inp):
        x = self.input_block(inp)
        x = self.down1(x)
        x = self.down2(x)
        x = self.resblocks(x)
        x = self.up1(x)
        x = self.up2(x)
        out = self.output_block(x)

        return out

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
    def __init__(self, input_ch, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        super(PatchDiscriminator, self).__init__()

        # No bias since BatchNorm2d has affine parameters
        if type(norm_layer) == functools.partial: 
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.Conv2d(input_ch, ndf, kernel_size=4, stride=2, padding=1),
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
    def __init__(self, input_ch, ndf=64, norm_layer=nn.BatchNorm2d):
        super(PixelDiscriminator, self).__init__()

        # No bias since BatchNorm2d has affine parameters
        if type(norm_layer) == functools.partial: 
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.Conv2d(input_ch, ndf, kernel_size=1, stride=1, padding=0),
                 nn.LeakyReLU(0.2, True),
                 nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
                 norm_layer(ndf * 2),
                 nn.LeakyReLU(0.2, True),
                 nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0)]
        
        self.model = nn.Sequential(*model)


    def forward(self, inp):
        return self.model(inp)