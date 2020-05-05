from tensorflow.keras.utils import Sequence
import numpy as np
import pandas as pd
import cv2


class PatchGenerator(Sequence):

    def __init__(self, info, image_path, mask_path, batch_size=32, dim=(128, 128), n_channels=3, shuffle=True):
        """
        :param list_IDs: list of all image labels (file names)
        :param image_path: path to images location
        :param mask_path: path to masks location
        :param to_fit: True to return X and y, False to return X only
        :param batch_size: batch size at each iteration
        :param dim: tuple indicating image dimension
        :param n_channels: number of image channels
        :param shuffle: True to shuffle label indexes after every epoch
        """
        self.info = info
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.shuffle = shuffle
        
        # Initialize indices
        self.indexes = np.arange(len(self.info))
        self.on_epoch_end()
        
        # Read images to save disk access
        img_names = info.img.unique().tolist()
        self.images = {}
        self.masks = {}
        for img_name in img_names:
            self.images[img_name] = cv2.imread(image_path + img_name + '.tif')
            self.masks[img_name] = cv2.imread(mask_path + img_name + '.png', cv2.IMREAD_GRAYSCALE)

    def __len__(self):
        """Denotes the number of batches per epoch

        :return: number of batches per epoch
        """
        return int(np.floor(len(self.info) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data

        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        X,y = self._generate_X_y(indexes)
        return X,y
      

    def on_epoch_end(self):
        """Updates indexes after each epoch

        """
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def _generate_X_y(self, list_IDs_temp):
        """Generates data containing batch_size images

        :param list_IDs_temp: list of label ids to load
        :return: batch of images
        """
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=np.uint8)
        y = np.empty((self.batch_size, *self.dim, 1), dtype=np.bool)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # augment according to info
            row = self.info.loc[ID]
            patch = self.images[row.img][row.c_stride : row.c_stride + self.dim[0] , row.r_stride : row.r_stride + self.dim[1]]
            mpatch = self.masks[row.img][row.c_stride : row.c_stride + self.dim[0] , row.r_stride : row.r_stride + self.dim[1]]
            
            if row.flip==True: 
                patch = cv2.flip(patch, 0)
                mpatch = cv2.flip(mpatch, 0)
            if row.rot==90:
                patch = cv2.rotate(patch, cv2.ROTATE_90_CLOCKWISE)
                mpatch = cv2.rotate(mpatch, cv2.ROTATE_90_CLOCKWISE)
            elif row.rot==180:
                patch = cv2.rotate(patch, cv2.ROTATE_180)
                mpatch = cv2.rotate(mpatch, cv2.ROTATE_180)
            elif row.rot==270:
                patch = cv2.rotate(patch, cv2.ROTATE_90_COUNTERCLOCKWISE)
                mpatch = cv2.rotate(mpatch, cv2.ROTATE_90_COUNTERCLOCKWISE)
                
            mpatch[mpatch!=0] = 1
            # Store sample
            X[i,] = patch
            y[i,] = np.expand_dims(mpatch, 2)

        return X, y
