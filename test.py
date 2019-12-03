import torch
import torch.nn as nn
import numpy as np
from imageio import imsave
import os

from metrics.ssim import ssim


def test(netG, loader, device, save_batches=0, save_path=None):
    with torch.no_grad(): 
        netG.eval()

        criterionMSE = nn.MSELoss().to(device)
        ssim_losses = []
        psrn_losses = []
        save_images_cleaned = []
        save_images_dirty = []
            
    
        for i, (real_inputs, real_targets, masks) in enumerate(loader):
            real_inputs, real_targets, masks = real_inputs.to(device), real_targets.to(device), masks.to(device)
            
            # Pass images through generator
            fake_targets = netG(torch.cat((real_inputs, masks), 1))

            # Get SSIM
            ssim_losses.append(ssim(fake_targets.detach(), real_targets).item())

            # Get PSRN 
            mse = criterionMSE(fake_targets, real_targets)
            psnr = 10 * np.log10(1 / mse.item())
            psrn_losses.append(psnr)

            # Save images
            if i < save_batches:
                for j in range(fake_targets.size(0)):
                    save_images_cleaned.append(((fake_targets[j].cpu().data.numpy().transpose(1, 2, 0)+1)*127.5).astype('uint8')) # Cleaned img
                    save_images_dirty.append(((real_targets[j].cpu().data.numpy().transpose(1, 2, 0)+1)*127.5).astype('uint8'))  # Corresponding dirty img

        avg_psrn = np.mean(psrn_losses)
        avg_ssim = np.mean(ssim_losses)
        
        # Save images to file
        if save_images:
            for i, img in enumerate(save_images_cleaned):
                imsave(os.path.join(save_path, f'clean_{i}.png'), img) 
            for i, img in enumerate(save_images_dirty):
                imsave(os.path.join(save_path, f'dirty_{i}.png'), img) 


        return avg_psrn, avg_ssim
