from .mask_refine import *

def load_network(hparams):
    if hparams.mode == 'mask_refine':
        if hparams.arch == 'encoder_decoder':
            return MREncoderDecoder(nf=hparams.nf, residual_blocks=hparams.n_blocks)
    else:
        raise NotImplementedError('')
