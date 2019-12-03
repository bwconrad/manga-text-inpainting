import torch
import torch.nn as nn

class GANLoss(nn.Module):
    def __init__(self, mode, real_label=1.0, fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(real_label))
        self.register_buffer('fake_label', torch.tensor(fake_label))
        self.mode = mode

        if mode == 'lsgan':
            self.loss = nn.MSLoss()
        elif mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        else:
            raise NotImplementedError('gan mode %s not implemented' % mode)

    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        if self.mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        #elif self.gan_mode == 'wgangp':
        #    if target_is_real:
        #        loss = -prediction.mean()
        #    else:
        #        loss = prediction.mean()
        return loss