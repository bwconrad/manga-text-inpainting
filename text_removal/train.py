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

def update_learning_rate(schedulers, type):
        '''
            Update learning rates for all the networks, called at the end of every epoch
        '''
        for scheduler in schedulers:
            if type == 'plateau':
                scheduler.step(0)
            else:
                scheduler.step()

def train_gan(netG, netD, vgg, train_loader, val_loader, optimizerG, optimizerD,
              schedulerG, schedulerD, criterionGAN, criterionL1, criterionPerceptual,
              criterionStyle, criterionTV, start_epoch, device, args, train_hist=None):
    
    if not train_hist:
        train_hist = {'D_losses': [], 
                      'G_losses': [], 
                      'L1_losses': [],
                      'Perceptual_losses': [],
                      'Style_losses': [],
                      'TV_losses': [], 
                      'PSRN': [], 
                      'SSIM': [],
                      'L1_val_losses': []}
    start = time.time()
    
    print('\nStarting to train...')
    for epoch in range(start_epoch, args.epochs+1):
        netG.train()
        netD.train()
        start_epoch = time.time()

        # Batch losses of the current epoch
        G_losses = [] 
        D_losses = []
        l1_losses = []
        perceptual_losses = []
        style_losses = []
        tv_losses = []

        for i, (real_inputs, real_targets, masks, edges) in enumerate(train_loader):
            real_inputs, real_targets, masks, edges = real_inputs.to(device), real_targets.to(device), masks.to(device), edges.to(device)
            fake_targets = netG(torch.cat((real_inputs, masks, edges), 1))

            ############
            # Update D #
            ############
            set_requires_grad(netD, True) # enable backprop for D
            optimizerD.zero_grad()

            # Real loss
            validity_real = netD(torch.cat((real_inputs, masks, edges), 1))
            d_real_loss = criterionGAN(validity_real, target_is_real=True)

            # Fake loss
            validity_fake = netD(torch.cat((fake_targets.detach(), masks, edges), 1))
            d_fake_loss = criterionGAN(validity_fake, target_is_real=False)

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
            validity_fake = netD(torch.cat((fake_targets, masks, edges), 1))  
            g_loss_gan = criterionGAN(validity_fake, target_is_real=True)                    
            
            # L1 loss - scaled by size of masked area
            g_loss_l1 = criterionL1(fake_targets, real_targets) / torch.mean(masks) if criterionL1 else 0 
             
            # Perceptual loss
            g_loss_perceptual = criterionPerceptual(fake_targets, real_targets, vgg) if criterionPerceptual else 0

            # Style loss
            g_loss_style = criterionStyle(fake_targets * masks, real_targets * masks, vgg) if criterionStyle else 0
            
            # Total variation loss
            g_loss_tv = criterionTV(fake_targets*masks + real_targets*(1-masks)) if criterionTV else 0

            # Combined loss
            g_loss = (g_loss_gan * args.lambda_gan) + (g_loss_l1 * args.lambda_l1) + \
                     (g_loss_perceptual * args.lambda_perceptual) + (g_loss_style * args.lambda_style) + \
                     (g_loss_tv * args.lambda_tv)
                     
            g_loss.backward()
            optimizerG.step()

            # Save batch losses
            D_losses.append(d_loss.detach().item())
            G_losses.append(g_loss_gan.detach().item())
            l1_losses.append(g_loss_l1.detach().item())
            perceptual_losses.append(g_loss_perceptual.detach().item())
            style_losses.append(g_loss_style.detach().item())
            tv_losses.append(g_loss_tv.detach().item())



            if (i+1)%args.batch_log_rate == 0:
                print('[Epoch {}/{}, Batch {}/{}] L1 loss: {:.6f} Perceptual loss: {:.6f} Style loss: {:.12f} TV loss: {:.6f}'
                      .format(epoch, args.epochs, i+1, len(train_loader), np.mean(l1_losses), 
                              np.mean(perceptual_losses), np.mean(style_losses), np.mean(tv_losses)))
            
        # Print epoch information
        print_epoch_stats(epoch, start_epoch, time.time(), D_losses, G_losses, l1_losses, perceptual_losses, style_losses, tv_losses, train_hist)
       
        # Evaluate on validation set
        print('Evaluating on validation set...')
        if epoch%args.save_samples_rate == 0:
            save_path = args.save_samples_path+'epoch{}/'.format(epoch)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            avg_psrn, avg_ssim, avg_val_l1 = test(netG, val_loader, device, save_batches=args.save_samples_batches, save_path=save_path)
        else:
            avg_psrn, avg_ssim, avg_val_l1 = test(netG, val_loader, device)
        train_hist['PSRN'].append(avg_psrn)
        train_hist['SSIM'].append(avg_ssim)
        train_hist['L1_val_losses'].append(avg_val_l1)
        print("PSRN: {} SSIM: {} L1: {}\n".format(avg_psrn, avg_ssim, avg_val_l1))
        
        # Save model
        save_checkpoint({'epoch': epoch,
                         'G_state_dict': netG.state_dict(),
                         'D_state_dict': netD.state_dict(),
                         'optimizerG_state_dict' : optimizerG.state_dict(),
                         'optimizerD_state_dict' : optimizerD.state_dict(),
                         'schedulerG_state_dict' : schedulerG.state_dict() if schedulerG else None,
                         'schedulerD_state_dict' : schedulerD.state_dict() if schedulerD else None,  
                         'args': args,
                         'train_hist': train_hist
                        }, epoch, args.checkpoint_path)

        
        # Save training history plot
        save_plots(train_hist, args.plot_path)

        # Update lr schedulers
        if schedulerG:
            update_learning_rate([schedulerG, schedulerD], args.scheduler)

    if args.archive:
        shutil.make_archive('images', 'zip', args.save_samples_path)
        shutil.make_archive('checkpoints', 'zip', args.checkpoint_path)
        shutil.make_archive('plots', 'zip', args.plot_path)


    hours, minutes, seconds = calculate_time(start, time.time())
    print('Training completed in {}h {}m {:04.2f}s'.format(hours, minutes, seconds))
        
