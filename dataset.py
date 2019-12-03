import torch
import torch.utils.data as data
from torchvision import transforms
from PIL import Image, ImageDraw
import numpy as np
import pandas as pd
import ast

def default_loader(path):
    return Image.open(path).convert('L')

class MangaDataset(data.Dataset):
    def __init__(self, data_root, ann_file, size = 256):
        # Load annotations
        print('Loading Annotations from {}'.format(ann_file))
        train_ann = pd.read_csv(data_root + ann_file)

        # Get file names and bounding boxes
        self.imgs = train_ann['file_name'].tolist()
        self.bboxes = train_ann['bounding_boxes'].tolist()
        print('\t {} Images'.format(len(self.imgs)))
        
        self.root = data_root
        self.loader = default_loader

        # Transformation parameters
        self.mean = [0.5]
        self.std = [0.5]
        self.size = size

        # Transformations
        self.tensor = transforms.ToTensor()
        self.norm = transforms.Normalize(mean=self.mean, std=self.std)
        self.resize = transforms.Resize(self.size)
        self.mask_resize = transforms.Resize(self.size, interpolation=0) # Have no antialiasing in resize 

    def create_mask(self, bboxes, w, h):
        # Create a black image of size (h,w)
        mask = Image.new('L', (w, h), color='black')
        draw = ImageDraw.Draw(mask)

        for (x0,y0), (x1,y1), _ in ast.literal_eval(bboxes):
            draw.rectangle([(x0,y0), (x1,y1)], fill='white', outline='white')

        return mask

        

    def __getitem__(self, index):
        dirty_path = self.root + 'dirty/' + self.imgs[index]
        dirty_img = self.loader(dirty_path)
        clean_path = self.root + 'clean/' + self.imgs[index]
        clean_img = self.loader(clean_path)

        bboxes = self.bboxes[index]
        mask = self.create_mask(bboxes, clean_img.size[0], clean_img.size[1]) # Create mask of text locations

        assert(clean_img.size[0]<clean_img.size[1]) # Make sure portrait image
        diff = clean_img.size[1] - clean_img.size[0]
        
        target_img = transforms.functional.pad(clean_img, padding=(int(np.floor(diff/2)), 0, int(np.ceil(diff/2)), 0)) # Add zero padding to make the image a square
        target_img = self.resize(target_img)
        target_img = self.tensor(target_img)
        target_img = self.norm(target_img)

        dirty_img = transforms.functional.pad(dirty_img, padding=(int(np.floor(diff/2)), 0, int(np.ceil(diff/2)), 0)) # Add zero padding to make the image a square
        dirty_img = self.resize(dirty_img)
        dirty_img = self.tensor(dirty_img)
        dirty_img = self.norm(dirty_img)

        mask = transforms.functional.pad(mask, padding=(int(np.floor(diff/2)), 0, int(np.ceil(diff/2)), 0)) # Add zero padding to make the image a square
        mask = self.mask_resize(mask)
        mask = self.tensor(mask)

        return dirty_img, target_img, mask

    def __len__(self):
        return len(self.imgs)