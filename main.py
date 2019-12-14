import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import sys
#from torchsummary import summary

from utils import *
from train import train_gan
from dataset import MangaDataset
from models import GlobalGenerator, VGG19
from loss import GANLoss, PerceptualLoss, StyleLoss, TVLoss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True
args = get_args()

# Load data
print('Loading data...')
train_dataset = MangaDataset(args.data_path + 'train/', 'train.csv', height=args.height, width=args.width)
val_dataset = MangaDataset(args.data_path + 'val/', 'val.csv', height=args.height, width=args.width)
test_dataset = MangaDataset(args.data_path + 'test/', 'test.csv', height=args.height, width=args.width)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

# Setup models
print('\nInitializing models...')
norm_layer = get_norm_layer(args.norm)
netG = GlobalGenerator(input_nc=2, output_nc=1, n_downsampling=args.n_downsamples_g, n_blocks=args.n_blocks_g, ngf=args.ngf, norm_layer=norm_layer,
                       use_dropout=args.dropout, use_spectral_norm=args.spectral_norm_g, dilation=args.dilation, kernel_size=args.kernel_size_g)
netG.to(device)
init_weights(netG, args.init_type, init_gain=args.init_gain)

netD = get_discriminator(args)
netD.to(device)
init_weights(netD, args.init_type, init_gain=args.init_gain)

if args.use_perceptual_loss or args.use_style_loss:
    vgg = VGG19()
    vgg.to(device)
else:
    vgg = None

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
criterionL1 = torch.nn.L1Loss() if args.use_l1_loss else None
criterionPerceptual = PerceptualLoss() if args.use_perceptual_loss else None
criterionStyle = StyleLoss() if args.use_style_loss else None
criterionTV = TVLoss() if args.use_tv_loss else None
print('Using losses: GAN={}, L1={}, Perceptual={}, Style={}, TV={}'
       .format(args.lambda_gan,
               args.lambda_l1 if criterionL1 else 0, 
               args.lambda_perceptual if criterionPerceptual else 0, 
               args.lambda_style if criterionStyle else 0,
               args.lambda_tv if criterionTV else 0))

# Train model
train_gan(netG, netD, vgg, train_loader, val_loader, optimizerG, optimizerD,
          schedulerG, schedulerD, criterionGAN, criterionL1, criterionPerceptual,
          criterionStyle, criterionTV, start_epoch, device, args, train_hist)



