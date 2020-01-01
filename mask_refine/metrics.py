import torch
import torch.nn as nn

class Accuracy(nn.Module):
    '''
        https://github.com/knazeri/edge-connect/blob/master/src/metrics.py
    '''
    def __init__(self, threshold=0.5):
        super(Accuracy, self).__init__()
        self.threshold = threshold

    def __call__(self, inputs, outputs):
        labels = (inputs > self.threshold)
        outputs = (outputs > self.threshold)

        relevant = torch.sum(labels.float())
        selected = torch.sum(outputs.float())

        if relevant == 0 and selected == 0:
            return torch.tensor(1), torch.tensor(1)

        true_positive = ((outputs == labels) * labels).float()
        recall = torch.sum(true_positive) / (relevant + 1e-8)
        precision = torch.sum(true_positive) / (selected + 1e-8)

        return precision, recall