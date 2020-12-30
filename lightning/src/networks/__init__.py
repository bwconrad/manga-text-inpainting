from .mask_refine import *

def load_network(hparams):
    if hparams.mode == 'mask_refine':
        if hparams.arch == 'encoder_decoder':
            net = MREncoderDecoder(nf=hparams.nf, residual_blocks=hparams.n_blocks)
            init_weights(net, hparams.weight_init, hparams.weight_gain)
        elif hparams.arch == 'u2net':
            net = U2Net()
    else:
        raise NotImplementedError('')

    return net

def init_weights(net, init_type='kaiming', gain=0.02):
        '''
        From: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/
        '''
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        print('Initializing weights as {}'.format(init_type.upper()))
        net.apply(init_func)