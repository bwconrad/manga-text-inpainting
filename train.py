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

def train_gan(netG, netD, train_loader, val_loader, optimizerG, optimizerD,
              schedulerG, schedulerD, criterionGAN, criterionL1, start_epoch, device, args):
    
    train_hist = {'D_losses': [], 'G_losses': [], 'L1_losses': [], 'PSRN': [], 'SSIM': []}
    start = time.time()
    
    print('\nStarting to train...')
    for epoch in range(start_epoch, args.epochs+1):
        netG.train()
        netD.train()
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

            # Real loss
            validity_real = netD(torch.cat((real_inputs, masks), 1))
            d_real_loss = criterionGAN(validity_real, target_is_real=True)

            # Fake loss
            validity_fake = netD(torch.cat((fake_targets.detach(), masks), 1))
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
            validity_fake = netD(torch.cat((fake_targets, masks), 1))  
            g_loss_gan = criterionGAN(validity_fake, target_is_real=True)                    
            
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
            
        # Save model
        save_checkpoint({'epoch': epoch,
                         'G_state_dict': netG.state_dict(),
                         'D_state_dict': netD.state_dict(),
                         'optimizerG_state_dict' : optimizerG.state_dict(),
                         'optimizerD_state_dict' : optimizerD.state_dict(),
                         'schedulerG_state_dict' : schedulerG.state_dict() if schedulerG else None,
                         'schedulerD_state_dict' : schedulerD.state_dict() if schedulerD else None,  
                         'args': args
                        }, epoch, args.checkpoint_path)

        # Print epoch information
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
        print("PSRN: {} SSIM: {}\n".format(avg_psrn, avg_ssim))
    
        # Save training history plot
        save_plots(train_hist, args.plot_path)

        # Update lr schedulers
        if schedulerG:
            update_learning_rate([schedulerG, schedulerD], args.scheduler)

    shutil.make_archive('images', 'zip', 'output/samples/')
    shutil.make_archive('checkpoints', 'zip', 'output/checkpoints/')

    hours, minutes, seconds = calculate_time(start, time.time())
    print('Training completed in {}h {}m {:04.2f}s'.format(hours, minutes, seconds))
        

