import pytorch_lightning as pl
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image, ImageDraw
import numpy as np
import pandas as pd
import ast
import os

def default_loader(path):
    return Image.open(path).convert('L')

class MaskRefineDataModule(pl.LightningDataModule): 
    def __init__(self, hparams):
        super(MaskRefineDataModule, self).__init__()
        self.hparams = hparams

    def setup(self, stage='fit'):
        if stage == 'fit':
            self.train_dataset = MaskRefineDataset(
                self.hparams.data_path, mode='train',
                height=self.hparams.height, width=self.hparams.width
            )
            self.val_dataset = MaskRefineDataset(
                self.hparams.data_path, mode='val',
                height=self.hparams.height, width=self.hparams.width
            )
        elif stage == 'test':
            self.test_dataset = MaskRefineDataset(
                self.hparams.data_path, mode='test',
                height=self.hparams.height, width=self.hparams.width
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, 
                          shuffle=True, num_workers=self.hparams.workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, 
                          shuffle=False, num_workers=self.hparams.workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, 
                          shuffle=False, num_workers=self.hparams.workers, pin_memory=True)

class MaskRefineDataset(data.Dataset): 
    def __init__(self, data_root, mode='train', height=256, width=256):
        super(MaskRefineDataset, self).__init__()

        self.mode = mode
        self.height = height
        self.width = width
        self.loader = default_loader

        # Set paths
        if mode == 'train':
            self.root = os.path.join(data_root, 'train/')
            ann_path = os.path.join(self.root, 'train.csv')
        elif mode == 'val':
            self.root = os.path.join(data_root, 'val/')
            ann_path = os.path.join(self.root, 'val.csv')
        else:
            self.root = os.path.join(data_root, 'test/')
            ann_path = os.path.join(self.root, 'test.csv')

        # Load annotation
        print(f'Loading Annotations from {ann_path}')
        ann = pd.read_csv(os.path.join(data_root, ann_path))

        # Get image paths and bounding boxes
        self.imgs = ann['file_name'].tolist()
        self.bboxes = ann['bounding_boxes'].tolist()
        print(f'\t Loading {len(self.imgs)} Images')

        # Tranformations
        self.tensor = transforms.ToTensor()
        self.norm = transforms.Normalize(mean=0.5, std=0.5)
        self.resize = transforms.Resize(min([height, width]))
        self.mask_resize = transforms.Resize(min([height, width]), interpolation=0) # NN resize


    def create_box_mask(self, bboxes, w, h, transform=False):
        # Create a black image of size (h,w)
        mask = Image.new('L', (w, h), color='black')
        draw = ImageDraw.Draw(mask)

        # Fill in box locations 
        for (x0,y0), (x1,y1), _ in ast.literal_eval(bboxes):
            if transform:
                # Randomly make box bigger
                x_off = np.random.uniform(high=(x1-x0)/2, size=2)
                y_off = np.random.uniform(high=(y1-y0)/4, size=2)
                x0, x1, y0, y1 = x0-x_off[0], x1+x_off[1], y0-y_off[0], y1+y_off[1]
            draw.rectangle([(x0,y0), (x1,y1)], fill='white', outline='white')

        return mask

    def __getitem__(self, index):
        # Load the image and text mask
        name = self.imgs[index]
        img = self.loader(os.path.join(self.root + 'dirty/' + name))
        mask_target = self.loader(os.path.join(self.root + 'mask_targets/' + name))

        # Create the input box mask
        bboxes = self.bboxes[index]
        mask_box = self.create_box_mask(bboxes, w=img.size[0], h=img.size[1], 
                                        transform=self.mode=='train') 
                
        # Apply transformations
        if self.mode != 'test':
            img = self.resize(img)
            mask_box = self.mask_resize(mask_box)
            mask_target = self.mask_resize(mask_target)
            if self.mode == 'train':
                [img, mask_box, mask_target] = random_crop_all([img, mask_box, mask_target], 
                                                               self.height, self.width)
            elif self.mode == 'val':
                [img, mask_box, mask_target] = center_crop_all([img, mask_box, mask_target], 
                                                               self.height, self.width)
        img = self.norm(self.tensor(img))
        mask_box = self.tensor(mask_box)
        mask_target = self.tensor(mask_target).round() # Loaded image is not binary

        return img, mask_box, mask_target, name

    def __len__(self):
        return len(self.imgs)


def random_crop_all(imgs, h, w):
    i, j, h, w = transforms.RandomCrop.get_params(
            imgs[0], output_size=(h, w))
    
    imgs_cropped = [] 
    for img in imgs:
        imgs_cropped.append(TF.crop(img, i, j, h, w))

    return imgs_cropped

def center_crop_all(imgs, height, width):
    imgs_cropped = [] 
    for img in imgs:
        imgs_cropped.append(TF.center_crop(img, (height, width)))

    return imgs_cropped

