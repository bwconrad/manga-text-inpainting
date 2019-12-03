import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import sys
from torchsummary import summary

from utils import *
from train import train_gan
from dataset import MangaDataset
from models import UNetGenerator, PatchDiscriminator, PixelDiscriminator, GlobalGenerator
from loss import GANLoss


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
print('\nSetting up models...')
norm_layer = get_norm_layer(args.norm)

netG = GlobalGenerator(input_nc=2, output_nc=1, n_downsampling=3, n_blocks=9, ngf=args.ngf, norm_layer=norm_layer)
netG.to(device)
init_weights(netG, args.init_type, init_gain=args.init_gain)

netD = PixelDiscriminator(input_ch=2, ndf=args.ndf, norm_layer=norm_layer)
netD.to(device)
init_weights(netD, args.init_type, init_gain=args.init_gain)

# Setup optimizer and scheduler
print('\nSetting up optimizer...')
optimizerG = optim.Adam(netG.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))
optimizerD = optim.Adam(netD.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))

schedulerG = get_scheduler(optimizerG, args)
schedulerD = get_scheduler(optimizerD, args)
print('Using a {} learning rate schedule'.format(type(schedulerG).__name__))


# Define loss functions
criterionGAN = GANLoss(args.gan_mode).to(device)
criterionL1 = torch.nn.L1Loss()

# Train model
train_gan(netG, netD, train_loader, val_loader, optimizerG, optimizerD,
          schedulerG, schedulerD, criterionGAN, criterionL1, device, args)


