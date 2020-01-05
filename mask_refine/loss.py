import torch
import torch.nn as nn

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
