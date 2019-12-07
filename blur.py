import pandas as pd 
import numpy as np 
import cv2 
import ast

path = './data/train/'
ann = pd.read_csv(path + 'train.csv')

imgs = ann['file_name'].tolist()
bboxes = ann['bounding_boxes'].tolist()

n = 10
file_name = imgs[n]

img = cv2.imread(path + 'dirty/' + file_name, cv2.IMREAD_GRAYSCALE)
clean = cv2.imread(path + 'dirty/' + file_name, cv2.IMREAD_GRAYSCALE)

for (x0,y0), (x1,y1), t in ast.literal_eval(bboxes[n]):
    if t == False:
        roi = img[y0:y1, x0:x1]
        print(roi.size)
        blur = cv2.GaussianBlur(roi, (171,5), 0)
        img[y0:y1, x0:x1] = blur
        print(x0,y0)


cv2.imshow('clean', clean)
cv2.imshow('blur', img)

cv2.waitKey(0) # waits until a key is pressed
