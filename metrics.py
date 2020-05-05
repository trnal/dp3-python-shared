import tensorflow as tf
from keras import backend as K

import cv2
import numpy as np
from vendor.lovasz_losses_tf import *


# Jaccard distance metric
# source: https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/losses/jaccard.py
# input y_true: ground truth (image, array-like)
#       y_pred: prediction (image, array-like) - same size as the ground truth
#       smooth: smoothing
# output jaccard distance - number (float)
def jaccard_distance(y_true, y_pred, smooth=100):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    
    return (1 - jac) * smooth


# Dice loss (implementation inspired by jaccard_distance)
# input y_true: ground truth (image, array-like)
#       y_pred: prediction (image, array-like) - same size as the ground truth
#       smooth: smoothing
# output dice loss - number (float)
def dice_loss(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1)
    dice = (2. * intersection + smooth) / (sum_ + smooth)
    
    return 1 - dice


def lovasz(y_true, y_pred):
    return lovasz_softmax(y_pred, y_true, classes=[1])


def dice_binary_crossentropy(y_true, y_pred):
    return dice_loss(y_true, y_pred) + K.binary_crossentropy(y_true, y_pred)

#
# ------------------- EVALUATION METRICS -----------------------
#


# Dice coefficient for probabilistic segmentation
# input A: ground truth (flattened image)
#       B: probabilistic prediction
# output: continuous dice coefficient
# source: rewritten from article: https://arxiv.org/pdf/1906.11031.pdf 
def continuous_dice_coefficient(A, B):
    size_of_A_intersect_B = sum(A * B)
    size_of_A = sum(A)
    size_of_B = sum(B)
    
    sign_B = np.zeros((B.shape))
    sign_B[B>0] = 1
    sign_B[B<0] = -1
    
    if (size_of_A_intersect_B > 0):
        c = sum(A*B)/sum(A*sign_B)
    else:
        c = 1
        
    return (2*size_of_A_intersect_B) / (c*size_of_A + size_of_B)


# Cell count accuracy
# input ground_truth: image (array-like) of uint8 background - 0, foreground - 255
#       mask: image to be evaluated (array-like) of uint8 background - 0, foreground - 255
#.      tolerance: threshold (0,1> criteria for accepting cell as sufficiently segmented
#.      k: constant <0,1> for penalizing sub-division of cells (when one cell in the ground 
#          truth is segmented asmultiple in mask
#       return raw: if true, returns TP, TN, FP, FN
#                   TP - number of cells in the ground truth which were segmented sufficiently 
#                   TN - 1 or 0 - 1 if at least of 50% background was segmented sufficiently 
#                   FP - number of cells in the mask which were not in the ground truth mask
#                   FN - number of cells in the ground truth which were not segmented sufficiently 
#                   FP_sub - number of extra cells that were in one cell area in the ground truth
# output cell count accuracy - floating number from <0,1>
#        TP, TN, FP, FN - tuple of four integer values (if return_raw=True)
def cell_count_accuracy(ground_truth, mask, tolerance=0.5, k=0.5, return_raw=False):
    
    if not tolerance > 0 or tolerance > 1:
        print('The threshold (tolerance) for accepting cells has to be a number in (0,1>')
        return
    
    no_cells, cells = cv2.connectedComponents(ground_truth)
    no_detected_cells, detected_cells = cv2.connectedComponents(mask)

    # 0/1 images - for multiplication purposes
    _ground_truth = ground_truth / 255
    _mask = mask / 255
    _intersection = _ground_truth * _mask

    tp = 0
    fn = 0
    fp_sub = 0

    for i in range(1, no_cells):
        # get cell
        cell = cells == i

        # find if at least 50% of the cell was detected
        if (cell * _intersection).sum() >= (cell.sum() * tolerance):
            tp += 1
        else:
            fn += 1

         # update mask and intersection (so that one cell is not counted multiple times)
        detected_cell_indices = np.unique(cell*detected_cells)
        _mask[np.isin(detected_cells, detected_cell_indices)] = 0
        _intersection = _ground_truth * _mask

        # find the extra cells in case of the cell sub-division (skip background - 0 value)
        detected_cell_indices = detected_cell_indices[detected_cell_indices != 0]
        if len(detected_cell_indices) > 1: fp_sub += (len(detected_cell_indices) -1)

    # find the extra remaining cells        
    remaining_cells = detected_cells * _mask
    fp = len(np.unique(remaining_cells)) - 1

    # background segmentation
    tn = 1 if (detected_cells == 0).sum() >= (cells == 0).sum() / 2 else 0

    if return_raw:
        return tp, tn, fp, fn, fp_sub

    return (tp + tn) / (tp + tn + fn + fp + k*fp_sub)
