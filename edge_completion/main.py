import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import sys

from utils import *
from dataset import EdgeMangaDataset
from models import EdgeGenerator, Discriminator
from loss import GANLoss, TverskyLoss
from train import train_gan

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True
args = get_args()

# Load data
print('Loading data...')
train_dataset = EdgeMangaDataset(args.data_path + 'train/', 'train.csv', height=args.height, width=args.width)
val_dataset = EdgeMangaDataset(args.data_path + 'val/', 'val.csv', height=args.height, width=args.width)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

# Setup models
print('\nInitializing models...')

netG = EdgeGenerator(ngf=args.ngf, residual_blocks=args.n_blocks_g, use_spectral_norm=args.spectral_norm_g, dilation=args.dilation)
netG.to(device)
init_weights(netG, args.init_type, init_gain=args.init_gain)

netD = Discriminator(in_channels=3, ndf=args.ndf, use_sigmoid=args.gan_loss != 'lsgan', use_spectral_norm=args.spectral_norm_d)
netD.to(device)
init_weights(netD, args.init_type, init_gain=args.init_gain)


# Setup optimizer and scheduler
print('\nSetting up optimizer...')
optimizerG = optim.Adam(netG.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))
optimizerD = optim.Adam(netD.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))

schedulerG, schedulerD = get_schedulers(optimizerG, optimizerD, args)

start_epoch = 1
train_hist = None

# Resume from checkpoint
if args.resume:
    print('\nLoading models from checkpoint {}'.format(args.resume))
    checkpoint = torch.load(args.resume)
    start_epoch = checkpoint['epoch']+1
    resume_args = checkpoint['args']
    train_hist = checkpoint['train_hist']

    netG.load_state_dict(checkpoint['G_state_dict'])
    netD.load_state_dict(checkpoint['D_state_dict'])

    optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
    optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])

    if checkpoint['schedulerG_state_dict']:
        schedulerG.load_state_dict(checkpoint['schedulerG_state_dict'])
        schedulerD.load_state_dict(checkpoint['schedulerD_state_dict'])

    print("Loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))

# Define loss functions
criterionGAN = GANLoss(args.gan_loss).to(device)
criterionFM = torch.nn.L1Loss() if args.use_fm_loss else None
criterionT = TverskyLoss(args.alpha_tversky, args.beta_tversky) if args.use_tversky_loss else None
print('Using losses: GAN={}, FM={}, Tversky={}'
       .format(args.lambda_gan,
               args.lambda_fm if criterionFM else 0,
               args.lambda_tversky if criterionT else 0))


train_gan(netG, netD, train_loader, val_loader, optimizerG, optimizerD,
          schedulerG, schedulerD, criterionGAN, criterionFM, criterionT, start_epoch,
          device, args, train_hist)