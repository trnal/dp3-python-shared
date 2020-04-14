from keras import backend as K

import cv2
import numpy as np


# Jaccard distance metric
# source: https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/losses/jaccard.py
# input y_true: ground truth (image, array-like)
#       y_pred: prediction (image, array-like) - same size as the ground truth
#       smooth: smoothing
# output jaccard distance number (float)
def jaccard_distance(y_true, y_pred, smooth=100):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth


#
# ------------------- EVALUATION METRICS -----------------------
#


# Cell count accuracy
# input ground_truth: image (array-like) of uint8 background - 0, foreground - 255
#       mask: image to be evaluated (array-like) of uint8 background - 0, foreground - 255
#       return raw: if true, returns TP, TN, FP, FN
#                   TP - number of cells in the ground truth which were in the mask (at least 50% of their area)
#                   TN - 1 or 0 - if at least of 50% background was detected correctly
#                   FP - number of cells in the mask which were not in the ground truth mask
#                   FN - number of cells in the ground truth which were not in the mask (at least 50% of their area)
# output cell count accuracy - floating number from <0,1>
#        TP, TN, FP, FN - four integer values (if return_raw=True)
def cell_count_accuracy(ground_truth, mask, return_raw=False):
    no_cells, cells = cv2.connectedComponents(ground_truth)
    no_detected_cells, detected_cells = cv2.connectedComponents(mask)

    # 0/1 images - for multiplication purposes
    _ground_truth = ground_truth / 255
    _mask = mask / 255
    _intersection = _ground_truth * _mask

    tp = 0
    fn = 0

    for i in range(1, no_cells):
        # get cell
        cell = cells == i

        # find if at least 50% of the cell was detected
        if (cell * _intersection).sum() >= (cell.sum() / 2):
            tp += 1
        else:
            fn += 1

        # update r and mr masks
        detected_cell_indices = np.unique(cell * detected_cells)
        _mask[np.isin(detected_cells, detected_cell_indices)] = 0
        _intersection = _ground_truth * _mask

    remaining_cells = detected_cells * _mask
    fp = len(np.unique(remaining_cells)) - 1

    tn = 1 if (detected_cells == 0).sum() >= (cells == 0).sum() / 2 else 0

    if return_raw:
        return tp, tn, fp, fn

    return (tp + tn) / (tp + fn + fp + 1)
