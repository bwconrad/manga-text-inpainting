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
              schedulerG, schedulerD, criterionGAN, criterionFM, start_epoch, 
              device, args, train_hist=None):
    
    if not train_hist:
        train_hist = {'D_losses': [], 
                      'G_losses': [], 
                      'FM_losses': [],
                      'Precision': [], 
                      'Recall': [],
                      'FM_val_losses': []}
    start = time.time()

    print('\nStarting to train...')
    for epoch in range(start_epoch, args.epochs+1):
        netG.train()
        netD.train()
        start_epoch = time.time()

        # Batch losses of the current epoch
        G_losses = [] 
        D_losses = []
        fm_losses = []
        
        for i, (images, masks, text_masks, edge_inputs, edge_targets, _) in enumerate(train_loader):
            images, masks, text_masks, edge_inputs, edge_targets = images.to(device), masks.to(device), text_masks.to(device), edge_inputs.to(device), edge_targets.to(device)
            fake_edge_outputs = netG(torch.cat((images, text_masks, edge_inputs), 1))

            ############
            # Update D #
            ############
            set_requires_grad(netD, True) # enable backprop for D
            optimizerD.zero_grad()

            # Real loss
            validity_real, dis_real_features = netD(torch.cat((images, text_masks, edge_targets), 1))
            d_real_loss = criterionGAN(validity_real, is_real=True, is_disc=True)

            # Fake loss
            validity_fake, dis_fake_features = netD(torch.cat((images, text_masks, fake_edge_outputs.detach()), 1))
            d_fake_loss = criterionGAN(validity_fake, is_real=False, is_disc=True)

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
            validity_fake, gen_fake_features = netD(torch.cat((images, text_masks, fake_edge_outputs), 1))  
            g_loss_gan = criterionGAN(validity_fake, is_real=True, is_disc=False)                    
            
            # Feature matching loss 
            g_loss_fm = 0
            for j in range(len(dis_real_features)):
                g_loss_fm += criterionFM(gen_fake_features[j], dis_real_features[j].detach())

            # Combined loss
            g_loss = (g_loss_gan * args.lambda_gan) + (g_loss_fm * args.lambda_fm)

            g_loss.backward()
            optimizerG.step()

            # Save batch losses
            D_losses.append(d_loss.detach().item())
            G_losses.append(g_loss_gan.detach().item())
            fm_losses.append(g_loss_fm.detach().item())

            if (i+1)%args.batch_log_rate == 0:
                print('[Epoch {}/{}, Batch {}/{}] FM loss: {}'
                      .format(epoch, args.epochs, i+1, len(train_loader), np.mean(fm_losses)))
            
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

        # Print epoch information
        print_epoch_stats(epoch, start_epoch, time.time(), D_losses, G_losses, fm_losses, train_hist)

        # Evaluate on validation set
        print('Evaluating on validation set...')
        if epoch%args.save_samples_rate == 0:
            save_path = args.save_samples_path+'epoch{}/'.format(epoch)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            avg_precision, avg_recall = test(netG, val_loader, device, save_batches=args.save_samples_batches, save_path=save_path)
        else:
            avg_precision, avg_recall = test(netG, val_loader, device)

        train_hist['Precision'].append(avg_precision)
        train_hist['Recall'].append(avg_recall)
        print("Precision: {} Recall: {}\n".format(avg_precision, avg_recall))

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