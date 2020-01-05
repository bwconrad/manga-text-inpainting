import torch
import torch.nn as nn

class GANLoss(nn.Module):
    '''
    Adversarial loss
    '''

    def __init__(self, type='vanilla', target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()

        self.type = type
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        if type == 'vanilla':
            self.criterion = nn.BCELoss()

        elif type == 'lsgan':
            self.criterion = nn.MSELoss()

        elif type == 'hinge':
            self.criterion = nn.ReLU()

    def __call__(self, outputs, is_real, is_disc=None):
        if self.type == 'hinge':
            if is_disc:
                if is_real:
                    outputs = -outputs
                return self.criterion(1 + outputs).mean()
            else:
                return (-outputs).mean()

        else:
            labels = (self.real_label if is_real else self.fake_label).expand_as(outputs)
            loss = self.criterion(outputs, labels)
            return loss

class TverskyLoss(nn.Module):
    '''
    Tversky loss
    https://arxiv.org/pdf/1706.05721.pdf
    '''

    def __init__(self, alpha, beta, eps=1e-8):
        '''
        Inputs:
            - alpha: false positive penalty
            - beta: false negative penalty
            - eps: add to demon for stability
        '''
        super(TverskyLoss, self).__init__()

        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def __call__(self, predictions, targets):
        # Get false positives/negatives and true positives
        fp = torch.sum(predictions * (1-targets))
        fn = torch.sum((1-predictions) * targets)
        tp = torch.sum(predictions * targets)

        num = tp 
        dem = tp + (self.alpha * fp) + (self.beta * fn) + self.eps 
        loss = 1 - ((num/dem))

        return loss