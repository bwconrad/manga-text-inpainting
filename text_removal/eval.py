import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
from imageio import imsave
import os
import sys

from metrics import ssim
from utils import *
from dataset import MangaDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True
args = get_args()

# Load data
print('Loading {} data...'.format(args.eval_dataset))
if args.eval_dataset == 'train':
    dataset = MangaDataset(args.data_path + 'train/', 'train.csv', height=args.height, width=args.width, edges=args.edges)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

elif args.eval_dataset == 'test':
    dataset = MangaDataset(args.data_path + 'test/', 'test.csv', height=args.height, width=args.width, edges=args.edges)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

elif args.eval_dataset == 'val':
    dataset = MangaDataset(args.data_path + 'val/', 'val.csv', height=args.height, width=args.width, edges=args.edges)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

else:
    print('dataset {} is not valid'.format(args.eval_dataset))
    sys.exit()


# Setup generator
print('\nInitializing model...')
netG = get_generator(args)
netG.to(device)

print('\nLoading model from checkpoint {}'.format(args.resume))
checkpoint = torch.load(args.resume)
resume_args = checkpoint['args']
netG.load_state_dict(checkpoint['G_state_dict'])
print("Loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))


print('Evaluating on {} dataset...'.format(args.eval_dataset))
save_path = './output/eval/' + args.eval_dataset + '/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

with torch.no_grad(): 
    netG.eval()

    criterionMSE = nn.MSELoss().to(device)
    criterionL1 = torch.nn.L1Loss()
    ssim_losses = []
    psrn_losses = []
    l1_losses = []
            
    # Measure psrn and ssim on entire val set
    for i, (inputs, targets, masks, text_masks, edges, names) in enumerate(loader):
        inputs, targets, masks, text_masks, edges = inputs.to(device), targets.to(device), masks.to(device), text_masks.to(device), edges.to(device)
            
        # Pass images through generator
        if args.generator == 'unet':           
            outputs = netG(torch.cat((inputs, text_masks, edges), 1), text_masks)
        else:
            outputs = netG(torch.cat((inputs, text_masks, edges), 1))

        # Get SSIM
        ssim_losses.append(ssim(outputs.detach(), targets).item())

        # Get PSRN 
        mse = criterionMSE(outputs, targets)
        psnr = 10 * np.log10(1 / mse.item())
        psrn_losses.append(psnr)

        # Get L1 loss
        l1 = criterionL1(outputs, targets) / torch.mean(masks)
        l1_losses.append(l1.detach().item())

        # Save images
        for j in range(outputs.size(0)):
            img = ((outputs[j].cpu().data.numpy().transpose(1, 2, 0)+1)*127.5).astype('uint8') 
            imsave(os.path.join(save_path, names[j]), img)

    avg_psrn = np.mean(psrn_losses)
    avg_ssim = np.mean(ssim_losses)
    avg_l1 = np.mean(l1_losses)

print('{} dataset PSRN: {} SSIM: {} L1 Loss: {}'.format(args.eval_dataset, avg_psrn, avg_ssim, avg_l1))