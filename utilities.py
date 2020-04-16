import cv2
import numpy as np
import math
import time
import pickle


def dump(obj, name, path, verbose=True):
    timestamp = str(time.time()).split('.')[0]
    classname = type(obj).__name__

    if classname == 'DataFrame':
        obj.to_csv(path + name + '__' + timestamp + '.csv')
    elif classname == 'dict':
        with open(path + name + '__' + timestamp + '.p', 'wb') as file_pi:
            pickle.dump(obj, file_pi)
    elif classname == 'Model':
        # store model in json and weights in .h5
        print('do something')
    else:
        with open(path + name + '__' + timestamp + '.txt', 'wb') as file_pi:
            file_pi.write(obj)

    if verbose:
        print('Written to: ' + path + name + '__' + timestamp)


def rotation_names():
    return ['', '_rot90', '_rot180', '_rot270', '_flip', '_flip_rot90', '_flip_rot_180', '_flip_rot270']


def rotate(img):
    result = np.zeros((8, *img.shape), dtype='uint8')
    result[0] = img
    result[1] = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    result[2] = cv2.rotate(img, cv2.ROTATE_180)
    result[3] = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    temp = cv2.flip(img, 0)
    result[4] = temp
    result[5] = cv2.rotate(temp, cv2.ROTATE_90_CLOCKWISE)
    result[6] = cv2.rotate(temp, cv2.ROTATE_180)
    result[7] = cv2.rotate(temp, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    return result


def rotate_back(rotated_images):
    imgs = np.squeeze(rotated_images)
    result = np.zeros(imgs.shape)
    result[0] = imgs[0]
    result[1] = cv2.rotate(imgs[1], cv2.ROTATE_90_COUNTERCLOCKWISE)
    result[2] = cv2.rotate(imgs[2], cv2.ROTATE_180)
    result[3] = cv2.rotate(imgs[3], cv2.ROTATE_90_CLOCKWISE)
    result[4] = cv2.flip(imgs[4], 0)
    result[5] = cv2.flip(cv2.rotate(imgs[5], cv2.ROTATE_90_COUNTERCLOCKWISE), 0)
    result[6] = cv2.flip(cv2.rotate(imgs[6], cv2.ROTATE_180), 0)
    result[7] = cv2.flip(cv2.rotate(imgs[7], cv2.ROTATE_90_CLOCKWISE), 0)
    
    return result


def find_last_patch_indices(img, stride, patch_width=128, patch_height=128):
    last_patch_vert = img.shape[0] - patch_height
    last_stride_vert = math.ceil((last_patch_vert) / stride) * stride
    last_patch_horiz = img.shape[1] - patch_width
    last_stride_horiz = math.ceil((last_patch_horiz) / stride) * stride
    
    return last_patch_vert, last_stride_vert, last_patch_horiz, last_stride_horiz
