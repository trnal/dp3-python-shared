import numpy as np
from utilities import *


# -------------------- COLLECTED OUTPUT -----------------------

def collect_output(model, img, stride=128, patch_size=128, padding=0):
    patch_sums = np.zeros((img.shape[0], img.shape[1]))
    patch_counts = np.zeros((img.shape[0], img.shape[1]))
    
    patch_pad = patch_size - padding
    stride = stride - 2*padding

    last_patch_vert, last_stride_vert, last_patch_horiz, last_stride_horiz = find_last_patch_indices(img, stride)

    i = 0
    while i + patch_size <= img.shape[0]:
        j = 0
        while j + patch_size <= img.shape[1]:
            
            xs = padding if i!= 0 else 0
            xe = patch_pad if i!=last_patch_vert else patch_size
            ys = padding if j!= 0 else 0
            ye = patch_pad if j!=last_patch_horiz else patch_size

            crop = img[i: i + patch_size, j: j + patch_size]
            predicted = model.predict(np.expand_dims(crop, axis=0))
            
            padded_prediction = np.squeeze(predicted)[xs : xe, ys : ye]
            
            patch_sums[i + xs : i + xe, j + ys : j + ye] += padded_prediction
            patch_counts[i + xs : i + xe, j + ys : j + ye] += 1

            j = j + stride
            if j == last_stride_horiz: j = last_patch_horiz

        i = i + stride
        if i == last_stride_vert: i = last_patch_vert

    return patch_sums / patch_counts


def collect_output_median(model, img, stride=128, patch_size=128, process_predicted_patch=None, predict_patch=None):
    predictions = np.ones((img.shape[0], img.shape[1], 1))*(-1)
    predictions = predictions.tolist()

    last_patch_vert, last_stride_vert, last_patch_horiz, last_stride_horiz = find_last_patch_indices(img, stride)

    i = 0
    while i + patch_size <= img.shape[0]:
        j = 0
        while j + patch_size <= img.shape[1]:

            crop = img[i: patch_size + i, j: patch_size + j]
            predicted = predict_patch(model, crop) if predict_patch else np.squeeze(model.predict(np.expand_dims(crop, axis=0)))
            if process_predicted_patch: predicted = process_predicted_patch(predicted)

            for row in range(i, i + patch_size):
                for col in range(j, j + patch_size): 
                    predictions[row][col].append(predicted[row-i][col-j]) 

            j = j + stride
            if j == last_stride_horiz: j = last_patch_horiz

        i = i + stride
        if i == last_stride_vert: i = last_patch_vert
            
    result = np.zeros((img.shape[0], img.shape[1]))
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            result[i][j] = np.median(predictions[i][j][1:])
    
    return result



def median_from_rotations(model, img):
    to_predict = rotate(img)
    predicted = model.predict(to_predict, verbose=0)
    back_rot = rotate_back(predicted)
    med = np.median(back_rot, axis=0)
    
    return np.squeeze(med)



# -------------------- BINARIZATION -----------------------------

def otsu_threshold(prediction):
    _prediction = __process_prediction(prediction)
    _, mask = cv2.threshold(_prediction, 0, 255, cv2.THRESH_OTSU)
    
    return mask


def constant_threshold(prediction, threshold=127):
    _prediction = __process_prediction(prediction)    
    _, mask = cv2.threshold(_prediction, 0, threshold, cv2.THRESH_BINARY)
    
    return mask


def adaptive_threshold_mean(prediction, block_size=11, c=2):
    _prediction = __process_prediction(prediction)
    mask = cv2.adaptiveThreshold(_prediction, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, c)
    
    return mask


def adaptive_threshold_gaussian(prediction, block_size=11, c=2):
    _prediction = __process_prediction(prediction)
    mask = cv2.adaptiveThreshold(_prediction, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, c)
    
    return mask


def __process_prediction(prediction):
    _prediction = np.squeeze(prediction *255)
    _prediction = _prediction.astype(dtype='uint8')
    
    return _prediction