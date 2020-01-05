import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import sys

from utils import *
from dataset import MaskRefineDataset
from models import Generator
from loss import TverskyLoss
from train import train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True
args = get_args()

# Load data
print('Loading data...')
train_dataset = MaskRefineDataset(args.data_path + 'train/', 'train.csv', height=args.height, width=args.width)
val_dataset = MaskRefineDataset(args.data_path + 'val/', 'val.csv', height=args.height, width=args.width)
#test_dataset = MaskRefineDataset(args.data_path + 'test/', 'test.csv', height=args.height, width=args.width)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
#test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

# Setup models
print('\nInitializing models...')

net = Generator(ngf=args.ngf, residual_blocks=args.n_blocks, use_spectral_norm=args.spectral_norm, dilation=args.dilation)
net.to(device)
init_weights(net, args.init_type, init_gain=args.init_gain)

# Setup optimizer and scheduler
print('\nSetting up optimizer...')
optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
scheduler = get_schedulers(optimizer, args)

start_epoch = 1
train_hist = None

# Resume from checkpoint
if args.resume:
    print('\nLoading models from checkpoint {}'.format(args.resume))
    checkpoint = torch.load(args.resume)
    start_epoch = checkpoint['epoch']+1
    resume_args = checkpoint['args']
    train_hist = checkpoint['train_hist']

    net.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    print("Loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))

# Define loss functions
criterionT = TverskyLoss(alpha=args.t_alpha, beta=args.t_beta) 
print('Using Tversky loss with Alpha={}, Beta={}'
       .format(args.t_alpha, args.t_beta ))

train(net, train_loader, val_loader, optimizer, scheduler, criterionT, start_epoch, device, args, train_hist)
