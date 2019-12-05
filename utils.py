import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler
from torchvision.utils import save_image
import functools
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os

from norm import *
from models import *

def init_weights(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__

        # Conv and lin layers
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

            # Biases
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

        # Batch norms
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)


    print('Initializing {} weights as {}'.format(type(net).__name__, init_type.upper()))
    net.apply(init_func)

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = lambda x: Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def get_scheduler(optimizer, args):
    if args.scheduler == 'linear':
        def lambda_rule(epoch):
            return 1.0 - max(0, epoch - args.start_lr_epochs) /float(args.decay_lr_epochs + 1)
        scheduler = lr_scheduler.LambdaLR(optimizer, lambda_rule)
    
    elif args.scheduler == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.start_lr_epochs, gamma=args.gamma)

    elif args.scheduler == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)

    elif args.scheduler == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)

    elif args.scheduler == 'none':
        scheduler = None

    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', args.scheduler)
    
    return scheduler

def get_args():
    ''' 
        Parse and return user arguments
    '''
    parser = argparse.ArgumentParser()

    # Path arguments
    parser.add_argument('--data_path', default='data/', help='path to data dirtectory')
    parser.add_argument('--save_samples_path', default='output/samples/', help='path to save samples')
    parser.add_argument('--checkpoint_path', default='output/checkpoints/', help='path to save checkpoints')
    parser.add_argument('--plot_path', default='output/plots/', help='path to save loss plots')
    parser.add_argument('--resume', default=None, help='checkpoint to resume training from')

    # Model arguments
    #parser.add_argument('--generator', required=True, help='unet | ')
    parser.add_argument('--discriminator', required=True, help='pixel | patch | multi')
    parser.add_argument('--width', type=int, default=512, help='width of input')
    parser.add_argument('--height', type=int, default=1024, help='height of input') 
    parser.add_argument('--gan_loss', type=str, default='vanilla', help='GAN loss function [vanilla | lsgan]')
    parser.add_argument('--norm', default='batch', help='normalization layer type [batch | instance | none]')
    parser.add_argument('--ngf', type=int, default=64, help='number of units in generator fully connected layers')
    parser.add_argument('--ndf', type=int, default=64, help='number of units in discriminator fully connected layers')
    parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal | xavier | kaiming | orthogonal]')
    parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
    parser.add_argument('--dropout', type=bool, default=True, help='use dropout')
    parser.add_argument('--spectral_norm', type=bool, default=False, help='use spectral normalization')
    parser.add_argument('--dilation', type=int, default=1, help='amount of dilation')
    parser.add_argument('--n_downsamples_g', type=int, default=3, help='# of downsamples in generator encoder')
    parser.add_argument('--n_blocks_g', type=int, default=9, help='# of resblocks in generator')
    parser.add_argument('--n_layers_d', type=int, default=3, help='# of layers in discriminator')
    parser.add_argument('--num_d', type=int, default=3, help='# of dicriminators in multiscale discriminator')



    # Optimization arguments
    parser.add_argument('--epochs', type=int, default=1, help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--lrD', type=float, default=0.001, help='discriminator learning rate')
    parser.add_argument('--lrG', type=float, default=0.001, help='generator learning rate')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for adam')
    parser.add_argument('--lambda_l1', type=float, default=100, help='lambda for L1 loss')
    parser.add_argument('--workers', type=int, help='number of workers', default=6) 
    
    # lr scheduler arguments
    parser.add_argument('--scheduler', default='none', help='learning rate schedule [linear | step | cosine | none]')
    parser.add_argument('--start_lr_epochs', type=int, default=10, help='# of epochs at starting learning rate')
    parser.add_argument('--decay_lr_epochs', type=int, default=10, help='# of epochs to linearly decay learning rate to zero')
    parser.add_argument('--step_lr_epochs', type=int, default=50, help='# of epochs between steps')
    parser.add_argument('--gamma', type=int, default=0.1, help='lr decay for step scheduling')
    
    # Logging arguments
    parser.add_argument('--batch_log_rate', type=int, default=50, help='update on training every number of batches')
    parser.add_argument('--save_samples_rate', type=int, default=1, help='save samples every number of epochs')
    parser.add_argument('--save_samples_batches', type=int, default=4, help='number of sample batches to save')


    return parser.parse_args()

def get_discriminator(args):
    norm_layer = get_norm_layer(args.norm)

    if args.discriminator == 'pixel':
        return PixelDiscriminator(input_ch=2, ndf=args.ndf, norm_layer=norm_layer)
    elif args.discriminator == 'patch':
        return PatchDiscriminator(input_ch=2, ndf=args.ndf, norm_layer=norm_layer, n_layers=args.n_layers_d)
    elif args.discriminator == 'multi':
        return MultiScaleDiscriminator(input_nc=2, ndf=args.ndf, norm_layer=norm_layer, n_layers=args.n_layers_d, num_D=args.num_d)
    else:
        raise NotImplementedError('discriminator [%s] is not implemented' % args.discriminator)




def calculate_time(start, end):
    '''
        Calculate and return the hours, minutes and seconds between start and end times
    '''
    hours, remainder = divmod(end-start, 60*60)
    minutes, seconds = divmod(remainder, 60)
    return int(hours), int(minutes), seconds

def save_checkpoint(state, epoch, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(state, save_path+'checkpoint_epoch{}.pth.tar'.format(epoch))

def print_epoch_stats(epoch, start, end, D_losses, G_losses, L1_losses, train_hist):
    # Save the average loss during the epoch
    avg_D_loss = np.mean(D_losses)
    train_hist['D_losses'].append(avg_D_loss)
        
    avg_G_loss = np.mean(G_losses)
    train_hist['G_losses'].append(avg_G_loss)

    avg_L1_loss = np.mean(L1_losses)
    train_hist['L1_losses'].append(avg_L1_loss)

    # Print epoch stats
    hours, minutes, seconds = calculate_time(start, end)
    print("\nEpoch {} Completed in {}h {}m {:04.2f}s: D loss: {} G loss: {} L1 loss: {}"
              .format(epoch, hours, minutes, seconds, avg_D_loss, avg_G_loss, avg_L1_loss))

def save_plots(train_hist, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # GAN Loss
    plt.figure(figsize=(10,5))
    plt.title("GAN Losses")
    plt.plot(train_hist['G_losses'],label="G")
    plt.plot(train_hist['D_losses'],label="D")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(save_path+'gan_loss.png')
    plt.cla()

    # L1 Loss
    plt.figure(figsize=(10,5))
    plt.title("L1 Losses")
    plt.plot(train_hist['L1_losses'],label="L1")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(save_path+'l1_loss.png')
    plt.cla()

    # SSIM
    plt.figure(figsize=(10,5))
    plt.title("Validation SSIM")
    plt.plot(train_hist['SSIM'],label="SSIM")
    plt.xlabel("Epoch")
    plt.ylabel("SSIM")
    plt.legend()
    plt.savefig(save_path+'ssim.png')
    plt.cla()

    # PSRN
    plt.figure(figsize=(10,5))
    plt.title("Validation PSRN")
    plt.plot(train_hist['PSRN'],label="PSRN")
    plt.xlabel("Epoch")
    plt.ylabel("PSRN")
    plt.legend()
    plt.savefig(save_path+'psrn.png')
    plt.cla()


