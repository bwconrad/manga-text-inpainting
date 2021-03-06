import torch
import torch.utils.data as data
from torchvision import transforms
from PIL import Image, ImageDraw
import numpy as np
import pandas as pd
import ast
from skimage import feature, img_as_float

def default_loader(path):
    return Image.open(path).convert('L')

class MaskRefineDataset(data.Dataset):
    def __init__(self, data_root, ann_file, height=1024, width=512):
        # Load annotations
        print('Loading Annotations from {}'.format(ann_file))
        ann = pd.read_csv(data_root + ann_file)

        # Get file names and bounding boxes
        self.imgs = ann['file_name'].tolist()
        self.bboxes = ann['bounding_boxes'].tolist()
        print('\t {} Images'.format(len(self.imgs)))
        
        self.root = data_root
        self.loader = default_loader

        # Transformation parameters
        self.mean = [0.5]
        self.std = [0.5]
        self.height = height
        self.width = width

        # Transformations
        self.tensor = transforms.ToTensor()
        self.norm = transforms.Normalize(mean=self.mean, std=self.std)
        self.resize = transforms.Resize((self.height, self.width))
        self.mask_resize = transforms.Resize((self.height, self.width), interpolation=0) # Have no antialiasing in resize 

    def create_mask(self, bboxes, w, h):
        # Create a black image of size (h,w)
        mask = Image.new('L', (w, h), color='black')
        draw = ImageDraw.Draw(mask)

        for (x0,y0), (x1,y1), _ in ast.literal_eval(bboxes):
            draw.rectangle([(x0,y0), (x1,y1)], fill='white', outline='white')

        return mask

    def __getitem__(self, index):
        # Load the dirty image
        name = self.imgs[index]
        dirty_path = self.root + 'dirty/' + name
        dirty_img = self.loader(dirty_path)
        
        # Load text mask
        text_path = self.root + 'mask_targets/' + name
        mask_text = self.loader(text_path) 

        # Create the mask
        bboxes = self.bboxes[index]
        mask = self.create_mask(bboxes, dirty_img.size[0], dirty_img.size[1]) # Create mask of text locations

        assert(dirty_img.size[0] < dirty_img.size[1]) # Make sure portrait image

        # Pad to 1:2 ratio
        if dirty_img.size[1] > 2*dirty_img.size[0]:
            # Pad width
            dw = (dirty_img.size[1] - 2*dirty_img.size[0]) / 2
            dirty_img = transforms.functional.pad(dirty_img, padding=(int(np.floor(dw)), 0,
                                                                      int(np.ceil(dw)), 0))            
            mask = transforms.functional.pad(mask, padding=(int(np.floor(dw)), 0,
                                                            int(np.ceil(dw)), 0))
            mask_text = transforms.functional.pad(mask_text, padding=(int(np.floor(dw)), 0,
                                                                      int(np.ceil(dw)), 0))    
            
        elif dirty_img.size[1] < 2*dirty_img.size[0]:
            # Pad height
            dh = (2*dirty_img.size[0] - dirty_img.size[1]) / 2
            dirty_img = transforms.functional.pad(dirty_img, padding=(0, int(np.floor(dh)), 
                                                                      0, int(np.ceil(dh))))
            mask = transforms.functional.pad(mask, padding=(0, int(np.floor(dh)), 
                                                            0, int(np.ceil(dh))))
            mask_text = transforms.functional.pad(mask_text, padding=(0, int(np.floor(dh)), 
                                                            0, int(np.ceil(dh))))

        # Apply transforms
        dirty_img = self.resize(dirty_img)
        dirty_img = self.tensor(dirty_img)
        dirty_img = self.norm(dirty_img)

        mask = self.mask_resize(mask)
        mask = self.tensor(mask)

        mask_text = self.mask_resize(mask_text)
        mask_text = self.tensor(mask_text)
        mask_text = torch.round(mask_text) # Fix problem pixels

        
        return dirty_img, mask, mask_text, name

    def __len__(self):
        return len(self.imgs)