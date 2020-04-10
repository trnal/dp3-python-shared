import os
import cv2
import math
from timeit import default_timer as timer

from utilities import rotate, rotation_names


class Augmentor:
    def __init__(self, patch_width, patch_height, stride, source_path, destination_path, destination_test_path='',
                 test_names=[], verbose=False):
        self.patch_width = patch_width
        self.patch_height = patch_height
        self.stride = stride
        self.source_path = source_path
        self.destination_path = destination_path
        self.destination_test_path = destination_test_path
        self.test_names = test_names
        self.verbose = verbose

    def generate_patches(self):
        img_names = [f for f in os.listdir(self.source_path) if not os.path.isdir(self.source_path + f)]
        if '.DS_Store' in img_names: img_names.remove('.DS_Store')

        if self.verbose: print('Starting generating patches from ' + str(len(img_names)) + ' images')

        for count, name in enumerate(img_names):
            i = 0
            img = cv2.imread(self.source_path + name)

            last_patch_vert = img.shape[0] - self.patch_height
            last_stride_vert = math.ceil((last_patch_vert) / self.stride) * self.stride
            last_patch_horiz = img.shape[1] - self.patch_width
            last_stride_horiz = math.ceil((last_patch_horiz) / self.stride) * self.stride

            start_image = timer()

            while i + self.patch_height <= img.shape[0]:
                j = 0
                k = 0
                if i == 0: start_row = timer()

                while j + self.patch_width <= img.shape[1]:

                    name_stem = os.path.splitext(name)[0]
                    new_name = name_stem + '_' + str(i) + '_' + str(j)
                    new_path = (self.destination_test_path if name_stem in self.test_names else self.destination_path) + new_name

                    # cropp
                    crop = img[i: self.patch_height + i, j: self.patch_width + j]
                    
                    # rotate
                    name_suffixes = rotation_names() 
                    rotated_crop = rotate(crop)
                    
                    # write 
                    for _idx, crop in enumerate(rotated_crop):
                        cv2.imwrite(new_path + name_suffixes[_idx] +'.png', crop)

                    k = k + 1
                    j = j + self.stride
                    if j == last_stride_horiz: j = last_patch_horiz

                i = i + self.stride
                if i == last_stride_vert: i = last_patch_vert
                if i == self.stride: end_row = timer()

            end_image = timer()
            if self.verbose and name == img_names[0]:
                print('One row with ' + str(k) + ' patches : time elapsed: ' + str(end_row - start_row))
            if self.verbose: print(
                '{0:02}'.format(count) + ' done: ' + name + '  :: time elapsed: ' + str(end_image - start_image))
