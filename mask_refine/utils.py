import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler
from torchvision.utils import save_image
import functools
import argparse
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})
import os


def get_args():
    ''' 
        Parse and return user arguments
    '''
    parser = argparse.ArgumentParser()

    # Path arguments
    parser.add_argument('--data_path', default='./data/', help='path to data directory')
    parser.add_argument('--save_samples_path', default='output/samples/', help='path to save samples')
    parser.add_argument('--checkpoint_path', default='output/checkpoints/', help='path to save checkpoints')
    parser.add_argument('--plot_path', default='output/plots/', help='path to save loss plots')
    parser.add_argument('--resume', default=None, help='checkpoint to resume training from')

    # Model arguments
    parser.add_argument('--width', type=int, default=512, help='width of input')
    parser.add_argument('--height', type=int, default=1024, help='height of input') 
    parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal | xavier | kaiming | orthogonal]')
    parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
    parser.add_argument('--n_blocks', type=int, default=8, help='# of resblocks')
    parser.add_argument('--spectral_norm', dest='spectral_norm', action='store_true', default=True, help='use spectral normalization')
    parser.add_argument('--dilation', type=int, default=2, help='amount of dilation')
    parser.add_argument('--ngf', type=int, default=64, help='# of feature channels in first layer')

    # Optimization arguments
    parser.add_argument('--epochs', type=int, default=1, help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for adam')
    parser.add_argument('--workers', type=int, help='number of workers', default=6) 

    # Loss functions arguments
    parser.add_argument('--t_alpha', type=float, default=0.1, help='alpha for tversky loss')
    parser.add_argument('--t_beta', type=float, default=0.9, help='beta for tversky loss')
        
    # lr scheduler arguments
    parser.add_argument('--scheduler', default='none', help='learning rate schedule [linear | step | cosine | none]')
    parser.add_argument('--start_lr_epochs', type=int, default=10, help='# of epochs at starting learning rate')
    parser.add_argument('--decay_lr_epochs', type=int, default=10, help='# of epochs to linearly decay learning rate to zero')
    parser.add_argument('--step_lr_epochs', type=int, default=50, help='# of epochs between steps')
    parser.add_argument('--step_gamma', type=int, default=0.1, help='lr decay for step scheduling')
    
    # Logging arguments
    parser.add_argument('--batch_log_rate', type=int, default=50, help='update on training every number of batches')
    parser.add_argument('--save_samples_rate', type=int, default=1, help='save samples every number of epochs')
    parser.add_argument('--save_samples_batches', type=int, default=4, help='number of sample batches to save')
    parser.add_argument('--archive', dest='archive', action='store_true', default=False, help='save samples and checkpoints as archives')


    args = parser.parse_args()

    if args.scheduler == 'linear':
        args.epochs = args.start_lr_epochs + args.decay_lr_epochs

    return args

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

def get_schedulers(optimizerG, args):
    if args.scheduler == 'linear':
        def lambda_rule(epoch):
            return 1.0 - max(0, epoch - args.start_lr_epochs) /float(args.decay_lr_epochs + 1)
        schedulerG = lr_scheduler.LambdaLR(optimizerG, lambda_rule)
        print('Using a LINEAR lr schedule starting with lr={} for {} epochs and decaying to 0 for {} epochs' \
               .format(args.lr, args.start_lr_epochs, args.decay_lr_epochs))
    
    elif args.scheduler == 'step':
        schedulerG = lr_scheduler.StepLR(optimizerG, step_size=args.step_lr_epochs, gamma=args.step_gamma)
        print('Using a STEP lr schedule starting with lr={} and decreasing by {} every {} epochs for {} epochs' \
               .format(args.lr, args.step_gamma, args.step_lr_epochs, args.epochs))

    elif args.scheduler == 'plateau':
        return NotImplementedError('Plateau lr schedule not implement yet')
        #scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
        

    elif args.scheduler == 'cosine':
        schedulerG = lr_scheduler.CosineAnnealingLR(optimizerG, T_max=args.epochs, eta_min=0)
        print('Using a COSINE lr schedule starting with lr={} for {} epochs' \
              .format(args.lr, args.epochs))

    elif args.scheduler == 'none':
        schedulerG = None
        print('Using NO lr schedule with lr={} for {} epochs' \
              .format(args.lr, args.epochs))

    else:
        return NotImplementedError('learning rate schedule {} is not implemented'.format(args.scheduler))
    
    return schedulerG

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

def print_epoch_stats(epoch, start, end, t_losses, train_hist):
    # Save the average loss during the epoch
    avg_t_loss = np.mean(t_losses)
    train_hist['T_losses'].append(avg_t_loss)

    # Print epoch stats
    hours, minutes, seconds = calculate_time(start, end)
    print("\nEpoch {} Completed in {}h {}m {:04.2f}s: Tversky loss: {}"
              .format(epoch, hours, minutes, seconds, avg_t_loss))

def save_plots(train_hist, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    x_range = np.arange(1, len(train_hist['T_losses'])+1)
  
    # Tversky Loss
    plt.figure(figsize=(10,5))
    plt.title("Tversky Losses")
    plt.plot(x_range, train_hist['T_losses'],label="T_train")
    plt.plot(x_range, train_hist['T_val_losses'],label="T_val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(save_path+'tversky_loss.png')
    plt.clf()

    # Precision
    plt.figure(figsize=(10,5))
    plt.title("Validation Precision")
    plt.plot(x_range, train_hist['Precision'],label="Precision")
    plt.xlabel("Epoch")
    plt.ylabel("Precision")
    plt.legend()
    plt.savefig(save_path+'precision.png')
    plt.clf()

    # Recall
    plt.figure(figsize=(10,5))
    plt.title("Validation Recall")
    plt.plot(x_range, train_hist['Recall'],label="Recall")
    plt.xlabel("Epoch")
    plt.ylabel("Recall")
    plt.legend()
    plt.savefig(save_path+'recall.png')
    plt.clf()


    # F1
    plt.figure(figsize=(10,5))
    plt.title("Validation F1")
    plt.plot(x_range, train_hist['F1'],label="F1")
    plt.xlabel("Epoch")
    plt.ylabel("F1")
    plt.legend()
    plt.savefig(save_path+'f1.png')
    plt.clf()
