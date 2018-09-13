"""
Utilities for real-time data augmentation on image data.
"""

import cv2
import numpy as np
import os

class SegmentationAugment(object):
    """
    segmentation augmentation

    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0,  # randomly shift images horizontally
    height_shift_range=0,  # randomly shift images vertically
    fill_mode='nearest',
    cval=0.,
    horizontal_flip=False,  # randomly flip images
    vertical_flip=False  # randomly flip images
    """

    def __init__(self, featurewise_center=False,
                 samplewise_center=False,
                 featurewise_std_normalization=False,
                 samplewise_std_normalization=False,
                 zca_whitening=False,
                 zca_epsilon=1e-6,
                 rotation_range=0,
                 width_shift_range=0.,
                 height_shift_range=0.,
                 brightness_range=None,
                 shear_range=0.,
                 zoom_range=0.,
                 channel_shift_range=0.,
                 fill_mode='nearest',
                 cval=0.,
                 horizontal_flip=False,
                 vertical_flip=False,
                 rescale=None,
                 preprocessing_function=None,
                 ):
        self.featurewise_center = featurewise_center
        self.samplewise_center = samplewise_center
        self.featurewise_std_normalization = featurewise_std_normalization
        self.samplewise_std_normalization = samplewise_std_normalization
        self.zca_whitening = zca_whitening
        self.zca_epsilon = zca_epsilon
        self.rotation_range = rotation_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.brightness_range = brightness_range
        self.shear_range = shear_range
        self.zoom_range = zoom_range
        self.channel_shift_range = channel_shift_range
        self.fill_mode = fill_mode
        self.cval = cval
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.rescale = rescale
        self.preprocessing_function = preprocessing_function

    def augment(self, np_inputs, np_masks):
        """
        Do image augment.
        Args:
            np_inputs: ndarray [B, H, W, C]
            np_masks: ndarray [B, H, W, Label]
        Return:
            image, label with augmentation will be return
        Raises:
            No
        """
        pair = zip(np_inputs, np_masks)
        pair_edit = [self.transform(img, mask) for img, mask in pair]

        np_inputs_new = [it[0] for it in pair_edit]
        np_labels_new = [it[1] for it in pair_edit]

        return np_inputs_new, np_labels_new
    
    def transform(self, image, label):
        """
        Wrap apply_transform_matrix, 
             get_transform_matrix,

        Args:
            image: [H,W,C]
            label: [H,W,C]
        Return:
            augmented image, label
        Raises:
            No
        """
        M = self.__get_transform_matrix(image)
        return self.__apply_transform_matrix(image,label, M)

    def __apply_transform_matrix(self, image, label, M):
        """
        Apply transform matrix to image and label
        Args:
            image: [H,W,C]
            label: [H,W,C]
            M: [3,3] image homography
        Return:
            augmented image, label
        Raises:
            No
        """
        h, w = image.shape[:2]
        image = cv2.warpPerspective(image, M, (w,h), flags=cv2.INTER_LINEAR)
        label = cv2.warpPerspective(label, M, (w,h), flags=cv2.INTER_NEAREST)
        return image, label

    def __get_transform_matrix(self, image):
        """
        get transform matrix (perspective matrix) [3x3]
        Args:
            image: [H,W,C]
        Return:
            an 3x3 matrix
        Raises:
            No
        """
        h, w = image.shape[:2]
        c_x = w / 2
        c_y = h / 2

        # transform matrix
        M = np.float32([[1,0,0],[0,1,0],[0,0,1]])

        # rotation
        if self.rotation_range != 0:
            angle = np.random.randint(-self.rotation_range, self.rotation_range+1)
            rotation_matrix = cv2.getRotationMatrix2D((c_x, c_y), angle, 1)
            rotation_matrix = np.vstack([rotation_matrix, [0,0,1]])
            M = np.dot(M, rotation_matrix)

        # zoom
        if self.zoom_range != 0:
            zx = 0.6 #np.random.rand(1) * self.zoom_range + 1
            zy = zx
            center_matrix = np.float32([[1,0,-c_x],[0,1,-c_y],[0,0,1]])
            M = np.dot(M, center_matrix)
            zoom_matrix = np.float32([[zx,0,0],[0,zy,0],[0,0,1]])
            M = np.dot(M, zoom_matrix)
            # center_matrix = np.float32([[1,0,c_x],[0,1,c_y],[0,0,1]])
            # M = np.dot(M, center_matrix)

        # translation
        if self.width_shift_range != 0 or self.height_shift_range != 0:
            tx = np.random.randint(-self.width_shift_range*w, self.width_shift_range*w+1)
            ty = np.random.randint(-self.height_shift_range*h, self.height_shift_range*h+1)
            translation_matrix = np.float32([[1,0,tx],[0,1,ty],[0,0,1]])
            M = np.dot(M, translation_matrix)

        return M

    ### legacy
    def __rotate(self, image, label):
        """
        Do image rotate
        Args:
            image: [H,W,C]
            label: [H,W,C] C: 1 or 3
        Return:
            an image and seg label will be returned
        Raises:
            No
        """
        h, w = image.shape[:2]
        c_x = w / 2
        c_y = h / 2
        angle = np.random.randint(-self.rotation_range, self.rotation_range)
        M = cv2.getRotationMatrix2D((c_x, c_y), angle, 1)
        image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR)
        label = cv2.warpAffine(label, M, (w, h), flags=cv2.INTER_NEAREST)
        
        return image, label
    
    ### legacy
    def __shift(self, image, label):
        """
        Do image translate
        Args:
            image: [H,W,C]
            label: [H,W,C] C: 1 or 3
        Return:
            an image and seg label will be returned
        Raises:
            No
        """

        h, w = image.shape[:2]
        c_x = w / 2
        c_y = h / 2
        tx = np.random.randint(-self.width_shift_range*w, self.width_shift_range*w)
        ty = np.random.randint(-self.height_shift_range*h, self.height_shift_range*h)
        M = np.float32([[1,0,tx],[0,1,ty]])
        image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR)
        label = cv2.warpAffine(label, M, (w, h), flags=cv2.INTER_NEAREST)

        return image, label
    
    ### legacy
    def __zoom(self, image, label):
        """
        Do image zoom
        Args:
            image: [H,W,C]
            label: [H,W,C] C: 1 or 3
        Return:
            an image and seg label will be returned
        Raises:
            No
        """
        h, w = image.shape[:2]
        
        zx = np.random.randint(-self.zoom_range*w, self.zoom_range*w)
        zy = np.random.randint(-self.zoom_range*h, self.zoom_range*h)
        M = np.float32([[zx,0,0],[0,zy,0]])
        image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR)
        label = cv2.warpAffine(label, M, (w, h), flags=cv2.INTER_NEAREST)

        return image, label