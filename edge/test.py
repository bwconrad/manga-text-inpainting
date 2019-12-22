import torch
import torch.nn as nn
import numpy as np
from imageio import imsave
import os

from metrics import EdgeAccuracy

def test(netG, loader, device, save_batches=0, save_path=None):
    with torch.no_grad(): 
        netG.eval()

        edgeacc = EdgeAccuracy().to(device)

        precisions = []
        recalls = []
        save_images_targets = []
        save_images_outputs = []

        # Measure psrn and ssim on entire val set
        for i, (images, masks, edge_inputs, edge_targets) in enumerate(loader):
            images, masks, edge_inputs, edge_targets = images.to(device), masks.to(device), edge_inputs.to(device), edge_targets.to(device)
            
            # Pass images through generator
            edge_outputs = netG(torch.cat((images, masks, edge_inputs), 1))

            # Get precision and recall
            precision, recall = edgeacc(edge_targets * masks, edge_outputs * masks)
            
            precisions.append(precision.item())
            recalls.append(recall.item())

            # Save images
            if i < save_batches:
                for j in range(edge_outputs.size(0)):
                    save_images_outputs.append(((edge_outputs[j].cpu().data.numpy().transpose(1, 2, 0)+1)*127.5).astype('uint8')) # Generated edge map
                    save_images_targets.append(((edge_targets[j].cpu().data.numpy().transpose(1, 2, 0)+1)*127.5).astype('uint8'))  # Corresponding corresponding ground truth edge map
            break
        avg_precision = np.mean(precisions)
        avg_recall = np.mean(recalls)

        # Save images to file
        if save_images_outputs:
            for i, img in enumerate(save_images_targets):
                imsave(os.path.join(save_path, f'target_{i}.png'), img) 
            for i, img in enumerate(save_images_outputs):
                imsave(os.path.join(save_path, f'output_{i}.png'), img) 

    return avg_precision, avg_recall