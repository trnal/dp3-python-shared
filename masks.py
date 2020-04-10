import math
import numpy as np


def collect_output(model, img, stride, patch_size=128):
    patch_sums = np.zeros(img.shape)
    patch_counts = np.zeros(img.shape)

    last_patch_vert = img.shape[0] - patch_size
    last_stride_vert = math.ceil((last_patch_vert) / stride) * stride
    last_patch_horiz = img.shape[1] - patch_size
    last_stride_horiz = math.ceil((last_patch_horiz) / stride) * stride

    i = 0
    
    while i + 128 <= img.shape[0]:
        j = 0
        while j + 128 <= img.shape[1]:
            
            crop = img[i: patch_size + i, j: patch_size + j]
            predicted = model.predict(np.expand_dims(crop, axis = 0))
            
            patch_sums[i: patch_size + i, j: patch_size + j]  += predicted[0]
            patch_counts[i: patch_size + i, j: patch_size + j]  += 1 

            j = j + stride
            if j == last_stride_horiz: j = last_patch_horiz

        i = i + stride
        if i == last_stride_vert: i = last_patch_vert
    
    return patch_sums/patch_counts
