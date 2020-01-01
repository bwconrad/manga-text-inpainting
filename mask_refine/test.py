import torch
import torch.nn as nn
import numpy as np
from imageio import imsave
import os

from metrics import Accuracy

def test(netG, loader, device, save_batches=0, save_path=None):
    with torch.no_grad(): 
        netG.eval()

        acc = Accuracy().to(device)

        precisions = []
        recalls = []
        save_images_targets = []
        save_images_outputs = []

        # Measure psrn and ssim on entire val set
        for i, (images, masks, text_masks, _) in enumerate(loader):
            images, masks, text_masks = images.to(device), masks.to(device), text_masks.to(device)
            
            # Pass images through generator
            text_mask_outputs = netG(torch.cat((images, masks), 1))

            # Get precision and recall
            precision, recall = acc(text_masks, text_mask_outputs)
            
            precisions.append(precision.item())
            recalls.append(recall.item())

            # Save images
            if i < save_batches:
                for j in range(text_mask_outputs.size(0)):
                    save_images_outputs.append(((text_mask_outputs[j].cpu().data.numpy().transpose(1, 2, 0))*255).astype('uint8')) # Generated refined mask
                    save_images_targets.append(((text_masks[j].cpu().data.numpy().transpose(1, 2, 0))*255).astype('uint8'))  # Corresponding corresponding ground truth edge map
            
        avg_precision = np.mean(precisions)
        avg_recall = np.mean(recalls)

        # Save images to file
        if save_images_outputs:
            for i, img in enumerate(save_images_targets):
                imsave(os.path.join(save_path, f'target_{i}.png'), img) 
            for i, img in enumerate(save_images_outputs):
                imsave(os.path.join(save_path, f'output_{i}.png'), img) 

    return avg_precision, avg_recall