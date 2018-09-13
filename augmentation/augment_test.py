import cv2
import numpy as np
from glob import glob
from keras.preprocessing.image import ImageDataGenerator
import random
import os
from augmentation import SegmentationAugment

data_path = './dataset/cityscape'

def obstacle_label(gt):
        human = np.where(gt==24,255,0) + np.where(gt==25,255,0)
        car = np.where(gt==26,255,0) + np.where(gt==27,255,0) + np.where(gt==28,255,0)
        road = np.where(gt==7,255,0) + np.where(gt==8,255,0)

        gt_new = road + car + human
        return gt_new

datagen_args = dict(
                fill_mode='nearest',
                cval=0.,
                horizontal_flip=False,
                vertical_flip=False)

aug_params = dict(rotation_range = 0.0,
                  width_shift_range = 0.,
                  height_shift_range = 0.,
                  zoom_range = 2
                 )
SegAug = SegmentationAugment(**aug_params)

image_datagen = ImageDataGenerator(**datagen_args)
mask_datagen = ImageDataGenerator(**datagen_args)

### generator
seed = random.randrange(1, 1000)
image_generator = image_datagen.flow_from_directory(
            os.path.join(data_path, 'train/IMAGE'),
            class_mode=None, seed=seed, batch_size=4, 
            target_size=(256,512),
            interpolation='nearest',
            color_mode='rgb')
mask_generator = mask_datagen.flow_from_directory(
            os.path.join(data_path, 'train/GT'),
            class_mode=None, seed=seed, batch_size=4, 
            target_size=(256,512),
            interpolation='nearest',
            color_mode='grayscale')

cv2.namedWindow('show', 0)
cv2.namedWindow('mask', 0)
for image, mask in zip(image_generator, mask_generator):

    image = image.astype(np.uint8)

    image, mask = SegAug.augment(image, mask)

    image = image[0]
    mask  = mask[0]
    cv2.imshow("mask", (mask.astype(np.float32)/9*255).astype(np.uint8))
    label = obstacle_label(mask).astype(np.uint8)

    img_color = cv2.cvtColor(image[:,:,1], cv2.COLOR_GRAY2BGR)
    label_color = cv2.applyColorMap(label, cv2.COLORMAP_JET)

    imgShow = cv2.addWeighted(img_color, 0.7, label_color, 0.5, 0.0)

    cv2.imshow('show', imgShow)
    key = cv2.waitKey()
    if key == 27:
        break
