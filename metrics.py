from keras import backend as K
import numpy as np


# https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/losses/jaccard.py
def jaccard_distance(y_true, y_pred, smooth=100):
  intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
  sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
  jac = (intersection + smooth) / (sum_ - intersection + smooth)
  return (1 - jac) * smooth


#
# ------------------- EVALUATION METRICS -----------------------
#


def cell_count_accuracy(ground_truth, mask, return_raw=False):
    
    no_cells, cells = cv2.connectedComponents(ground_truth)
    no_det_cells, det_cells = cv2.connectedComponents(mask)
    
    m = ground_truth/255
    r = mask/255 
    mr = m * r
    
    tp = 0
    fn = 0

    for i in range(1, no_cells):
        # get cell
        cell = cells == i

        # find if at least 50% of the cell was detected
        if (cell * mr).sum() >= (cell.sum()/2):
            tp += 1
        else:
            fn += 1

        # update r and mr masks
        det_cell_vals = np.unique(cell*det_cells)
        r[np.isin(det_cells, det_cell_vals)] = 0
        mr = m * r

    remaining_cells = det_cells * r
    fp = len(np.unique(remaining_cells)) - 1
    
    tn = 1 if (det_cells==0).sum() >= (cells==0).sum()/2 else 0
    
    if return_raw: return tp, tn, fp, fn

    return (tp + tn) / (tp + fn + fp + 1)
