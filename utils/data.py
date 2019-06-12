"""
data generator and augmentation
"""
import argparse
import os
import random
from glob import glob

import numpy as np
import cv2
from keras.preprocessing.image import ImageDataGenerator
from scipy.misc import imresize
from scipy.ndimage import imread
from skimage import color
from matplotlib import pyplot as plt

seed = 1

def _motion_blur(img):
    seed = random.randint(0, 5)
    if seed == 0:
        size = 15
        kernel_motion_blur = np.zeros((size, size))
        kernel_motion_blur[(size-1) // 2, :] = np.ones(size)
        kernel_motion_blur = kernel_motion_blur / size
        return cv2.filter2D(img, -1, kernel_motion_blur)
    else:
        return img

def _mask_motion_blur(img):
    img[np.where(img<0.9)] = 0
    img[np.where(img>=0.9)] = 1
    return img

def _create_datagen(img_file, mask_file, img_gen, mask_gen, batch_size, target_size):

    img_iter = img_gen.flow_from_directory(img_file, class_mode=None, color_mode="rgb", batch_size=batch_size, target_size = target_size, seed=seed)
    mask_iter = mask_gen.flow_from_directory(mask_file, class_mode=None, color_mode="grayscale", batch_size=batch_size, target_size = target_size,
                              # use same seed to apply same augmentation with image and mask
                              seed=seed)
    datagen = zip(img_iter, mask_iter)

    return datagen

def load_data(train_dir, val_dir, batch_size, target_size, featurewise=False):
    
    # feature wise standardlization
    if featurewise:
        pass
    # sample wise standardlization
    else:
        # train images and masks generator
        train_img_gen = ImageDataGenerator(
            samplewise_center=True,
            rescale= 1./255,
            samplewise_std_normalization=True,
            rotation_range=20,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            preprocessing_function=_motion_blur,
        )
        train_mask_gen = ImageDataGenerator(
            rescale= 1./255,
            rotation_range=20,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            # preprocessing_function=_mask_motion_blur,
        )    
        train_gen = _create_datagen(
            train_dir+'/images',
            train_dir+'/masks',
            img_gen = train_img_gen,
            mask_gen = train_mask_gen,
            batch_size = batch_size,
            target_size = target_size
        )

        # val images and masks generator
        validation_img_gen = ImageDataGenerator(
            rescale = 1./255,
            samplewise_center=True,
            samplewise_std_normalization=True,
            horizontal_flip=True,
        )
        validation_mask_gen = ImageDataGenerator(
            rescale= 1./255,
            horizontal_flip=True,
        )
        validation_gen = _create_datagen(
            val_dir+'/images/',
            val_dir+'/masks/',
            img_gen = validation_img_gen,
            mask_gen = validation_mask_gen,
            batch_size=batch_size, 
            target_size=target_size
        )        
    return train_gen, validation_gen


if __name__ == '__main__':

    '''debug utility to show augumented images
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train_dir',
        type=str,
        default='/home/ubuntu/ihandy_seg/data/hair/CelebA/train',
        help='directory in which images and masks are placed.'
    )
    parser.add_argument(
        '--val_dir',
        type=str,
        default='/home/ubuntu/ihandy_seg/data/hair/CelebA/train',
        help='directory to put outputs.'
    )
    parser.add_argument(
        '--target_size',
        type=int,
        default=(256, 192),
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='batch size',
    )
    parser.add_argument(
        '--featurewise',
        type=bool,
        default=False,
        help='use samplewise or featurewise standardlization',
    )

    args, _ = parser.parse_known_args()
    train_gen, val_gen = load_data(**vars(args))
    
    # show the images for easy debug
    while True:
        for img, mask in train_gen:
            plt.subplot(1,2,1)
            plt.imshow(img[3])
            plt.subplot(1,2,2)
            plt.imshow(mask[3].reshape(mask.shape[1:3]))
            plt.show()


