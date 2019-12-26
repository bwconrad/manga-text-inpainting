import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
from imageio import imsave
import os

from metrics import EdgeAccuracy
from utils import *
from dataset import EdgeMangaDataset
from models import EdgeGenerator, Discriminator


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True
args = get_args()

# Load data
print('Loading data...')
train_dataset = EdgeMangaDataset(args.data_path + 'train/', 'train.csv', height=args.height, width=args.width)
val_dataset = EdgeMangaDataset(args.data_path + 'val/', 'val.csv', height=args.height, width=args.width)
test_dataset = EdgeMangaDataset(args.data_path + 'test/', 'test.csv', height=args.height, width=args.width)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

# Setup models
print('\nInitializing models...')

netG = EdgeGenerator(ngf=args.ngf, residual_blocks=args.n_blocks_g, use_spectral_norm=args.spectral_norm_g, dilation=args.dilation)
netG.to(device)

netD = Discriminator(in_channels=3, ndf=args.ndf, use_sigmoid=args.gan_loss != 'lsgan', use_spectral_norm=args.spectral_norm_d)
netD.to(device)

print('\nLoading models from checkpoint {}'.format(args.resume))
checkpoint = torch.load(args.resume)
resume_args = checkpoint['args']
netG.load_state_dict(checkpoint['G_state_dict'])
netD.load_state_dict(checkpoint['D_state_dict'])
print("Loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))

for i, (dataset, loader) in enumerate([('val', val_loader), ('train', train_loader), ('test', test_loader)]):
    print('Evaluating on {} dataset...'.format(dataset))
    save_path = './' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    with torch.no_grad(): 
        netG.eval()

        edgeacc = EdgeAccuracy().to(device)

        precisions = []
        recalls = []
        save_images_targets = []
        save_images_outputs = []

        # Measure psrn and ssim on entire val set
        for i, (images, masks, edge_inputs, edge_targets, names) in enumerate(loader):
            images, masks, edge_inputs, edge_targets = images.to(device), masks.to(device), edge_inputs.to(device), edge_targets.to(device)
            
            # Pass images through generator
            edge_outputs = netG(torch.cat((images, masks, edge_inputs), 1))

            # Get precision and recall
            precision, recall = edgeacc(edge_targets * masks, edge_outputs * masks)
            
            precisions.append(precision.item())
            recalls.append(recall.item())

            # Save images
            for j in range(edge_outputs.size(0)):
                img = ((edge_outputs[j].cpu().data.numpy().transpose(1, 2, 0))*255).astype('uint8') # Generated edge map
                imsave(os.path.join(save_path, names[j]), img)

        avg_precision = np.mean(precisions)
        avg_recall = np.mean(recalls)

    print('{} dataset precision: {} recall: {}'.format(dataset, avg_precision, avg_recall))