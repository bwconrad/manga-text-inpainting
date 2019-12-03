import torch
import time
import os
import shutil

from utils import *
from test import *

def set_requires_grad(nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

def train_gan(netG, netD, train_loader, val_loader, optimizerG, optimizerD,
              schedulerG, schedulerD, criterionGAN, criterionL1, device, args):
    netG.train()
    netD.train()
    train_hist = {'D_losses': [], 'G_losses': [], 'L1_losses': [], 'PSRN': [], 'SSIM': []}
    start = time.time()
    
    print('\nStarting to train...')
    for epoch in range(1, args.epochs+1):
        start_epoch = time.time()

        # Batch losses of the current epoch
        G_losses = [] 
        D_losses = []
        L1_losses = []

        for i, (real_inputs, real_targets, masks) in enumerate(train_loader):
            real_inputs, real_targets, masks = real_inputs.to(device), real_targets.to(device), masks.to(device)
            fake_targets = netG(torch.cat((real_inputs, masks), 1))

            ############
            # Update D #
            ############
            set_requires_grad(netD, True) # enable backprop for D
            optimizerD.zero_grad()

            # Real
            validity_real = netD(torch.cat((real_inputs, masks), 1))
            d_real_loss = criterionGAN(validity_real, target_is_real=True)

            # Fake
            validity_fake = netD(torch.cat((fake_targets.detach(), masks), 1))
            d_fake_loss = criterionGAN(validity_real, target_is_real=False)

            # Combined loss
            d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss.backward()
            optimizerD.step()

            ############
            # Update G #
            ############
            set_requires_grad(netD, False)  # D requires no gradients when optimizing G
            optimizerG.zero_grad()   

            # GAN loss
            validity_fake = netD(torch.cat((fake_targets, masks), 1))  
            g_loss_gan = criterionGAN(validity_fake, True)                    
            
            # L1 loss
            g_loss_l1 = criterionL1(fake_targets, real_targets) 

            # Combined loss
            g_loss = g_loss_gan + (g_loss_l1 * args.lambda_l1)
            g_loss.backward()
            optimizerG.step()

            # Save batch losses
            D_losses.append(d_loss.detach().item())
            G_losses.append(g_loss_gan.detach().item())
            L1_losses.append(g_loss_l1.detach().item())

            if (i+1)%args.batch_log_rate == 0:
                print('[Epoch {}, Batch {}/{}] L1 loss: {:.6f}'.format(epoch, i+1, len(train_loader), np.mean(L1_losses)))
            break
        '''
        Save model
        '''

        print_epoch_stats(epoch, start_epoch, time.time(), D_losses, G_losses, L1_losses, train_hist)
        
        # Evaluate on validation set
        print('Evaluating on validation set...')
        
        if epoch%args.save_samples_rate == 0:
            save_path = args.save_samples_path+'epoch{}/'.format(epoch)
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            avg_psrn, avg_ssim = test(netG, val_loader, device, save_batches=args.save_samples_batches, save_path=save_path)
        
        else:
            avg_psrn, avg_ssim = test(netG, val_loader, device)
        
        train_hist['PSRN'].append(avg_psrn)
        train_hist['SSIM'].append(avg_ssim)
        print("PSRN: {} SSIM: {}".format(avg_psrn, avg_ssim))

        break
        # Save training history plot
        save_loss_plot(train_hist['G_losses'], train_hist['D_losses'], train_hist['L1_losses'], epoch, args.plot_path+'loss/')
        save_metrics_plot(train_hist['PSRN'], train_hist['SSIM'], epoch, args.plot_path+'metrics/')
        
    shutil.make_archive('images', 'zip', 'manga-text-inpainting/output/samples/')

        

