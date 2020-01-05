from skimage import io, feature, img_as_float
from skimage.color import rgb2gray
from skimage.transform import resize, downscale_local_mean
from skimage.feature import canny
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import sys
from PIL import Image, ImageDraw
import ast

def create_mask( bboxes, w, h):
    # Create a black image of size (h,w)
    mask = Image.new('L', (w, h), color='black')
    draw = ImageDraw.Draw(mask)

    for (x0,y0), (x1,y1), _ in ast.literal_eval(bboxes):
        draw.rectangle([(x0,y0), (x1,y1)], fill='white', outline='white')

    return mask


root = '~/code/ml/manga-text-remover/data/train/'
ann = pd.read_csv(root + 'train.csv')
imgs = ann['file_name'].tolist()
bboxes = ann['bounding_boxes'].tolist()

input_dir = './edge/train/inputs/'
target_dir = './edge/train/targets/'


for i in range(len(imgs)):
    i = 2
    img = rgb2gray(io.imread(root + 'dirty/' + imgs[i]))
    img = img_as_float(img) # Convert to [0,1]
    #img = rescale(img, (512, 256))
    boxes = bboxes[i]
    #mask = create_mask(boxes, img.shape[1], img.shape[0])
    mask = rgb2gray(io.imread(root + 'mask/' + imgs[i]))
    mask = np.round(img_as_float(mask)) # Convert to [0,1]

    print(np.max(img))
    edges = canny(img, sigma=2, low_threshold=0.05) # Get edges
    #mask_edges = img * (1-mask)

    #plt.imsave(input_dir + imgs[i], edges, cmap = "gray")
    #plt.imsave(target_dir + imgs[i], mask_edges, cmap = "gray")
    
    print(i)
    
    fig = plt.figure(figsize=(4,4))

    #fig.add_subplot(1,4,1)
    #plt.imshow( mask_edges, cmap='gray')
    fig.add_subplot(1,4,2)
    plt.imshow( edges, cmap='gray')
    fig.add_subplot(1,4,3)
    plt.imshow( mask, cmap='gray')
    fig.add_subplot(1,4,4)
    plt.imshow( img, cmap='gray')

    plt.show()
    sys.exit()
    