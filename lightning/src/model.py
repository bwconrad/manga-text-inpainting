import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.utils import make_grid
import pytorch_lightning as pl 
from pytorch_lightning.metrics.classification import Precision, Recall
import numpy as np
from collections import OrderedDict

from .networks import load_network
from .loss import TverskyLoss
from .utils import get_scheduler 


class MaskRefineModel(pl.LightningModule):
    def __init__(self, hparams):
        super(MaskRefineModel, self).__init__()
        self.save_hyperparameters()

        # Hyperparameters
        self.hparams = hparams
        self.lr = hparams.lr
        
        # Modules
        self.net = load_network(hparams)
        self.criterion = TverskyLoss(hparams.tversky_alpha, hparams.tversky_beta)
        
        # Metrics
        self.train_precision = Precision()
        self.train_recall = Recall()
        self.val_precision = Precision()
        self.val_recall = Recall()

    def forward(self, img, bboxes):
        return self.net(torch.cat((img, bboxes), 1))

    def training_step(self, batch, batch_idx):
        img, bboxes, target, _ = batch

        # Pass through model
        pred = self(img, bboxes)

        # Calculate loss and metrics
        loss = self.criterion(pred, target)
        precision = self.train_precision(pred, target)
        recall = self.train_recall(pred, target)

        self.log('train_loss', loss)
        self.log('train_prec', precision)
        self.log('train_rec', recall)
        self.log('train_f1', (2*precision*recall)/(precision+recall))
        self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'], prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        img, bboxes, target, _ = batch

        # Pass through model
        pred = self(img, bboxes)

        # Calculate loss and metrics
        loss = self.criterion(pred, target)
        precision = self.val_precision(pred, target)
        recall = self.val_recall(pred, target)
        
        self.log('val_loss', loss)
        self.log('val_prec', precision, on_epoch=True)
        self.log('val_rec', recall, on_epoch=True)
        self.log('val_f1', (2*precision*recall)/(precision+recall), on_epoch=True)
        
        return OrderedDict({
            'val_loss': loss,
            'val_prec': precision,
            'val_rec': recall,
            'val_sample': torch.stack([img[0], target[0], pred[0]]) \
                if batch_idx%(250//self.hparams.batch_size) == 0 else None # Save some samples for visualization
        })

    def validation_epoch_end(self, outputs):
        # Visualize validation samples
        imgs = torch.cat([x['val_sample'] for x in outputs if x['val_sample'] is not None], 0)
        grid = make_grid(imgs, nrow=3, normalize=True)
        
        # Log to tensorboard
        tensorboard = self.logger.experiment[0]
        tensorboard.add_image('val_samples', grid, self.current_epoch)

        # Print validation results
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_prec = torch.stack([x['val_prec'] for x in outputs]).mean()
        avg_rec = torch.stack([x['val_rec'] for x in outputs]).mean()
        avg_f1 = (2*avg_prec*avg_rec)/(avg_prec+avg_rec) 
        print(f'\nVal Loss: {avg_loss:.5f} Val Precision: {avg_prec:.4f}',
                f'Val Recall: {avg_rec:.4f} Val F1 {avg_f1:.4f}')


    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, 
                               betas=(self.hparams.beta1, self.hparams.beta2))
        if self.hparams.schedule != 'none':
            scheduler = get_scheduler(optimizer, self.lr, self.hparams)
            return [optimizer], [scheduler]
        else:
            print(f'Using no LR schedule lr={self.lr}')
            return [optimizer]