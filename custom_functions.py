from keras.models import Model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate

import pandas as pd
import numpy as np
import cv2
import os
import math

PATCH_WIDTH = 128
PATCH_HEIGHT = 128
PATCH_CHANNELS = 3


# MODEL ARCHITECTURE

def base_model(loss='binary_crossentropy', optimizer='adam', verbose=True):
    inputs = Input((PATCH_HEIGHT, PATCH_WIDTH, PATCH_CHANNELS))
    s = Lambda(lambda x: x / 255)(inputs)

    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(s)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D((2, 2))(c4)

    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c5)

    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer=optimizer, loss=loss)

    if verbose: model.summary()

    return model


# create folder
def create_folder(path):
    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed" % path)


# compare predictions 
def predict_all_classes(img, model_epi, model_lym, model_mac, model_neu):
    pred_result_epi = collect_output(model_epi, img, stride=32, padding=8)
    pred_result_lym = collect_output(model_lym, img, stride=32, padding=8)
    pred_result_mac = collect_output(model_mac, img, stride=32, padding=8)
    pred_result_neu = collect_output(model_neu, img, stride=32, padding=8)

    epi_mask = np.ones(pred_result_epi.shape)
    epi_mask[pred_result_epi < pred_result_lym] = 0
    epi_mask[pred_result_epi < pred_result_neu] = 0
    epi_mask[pred_result_epi < pred_result_mac] = 0

    lym_mask = np.ones(pred_result_epi.shape)
    lym_mask[pred_result_lym < pred_result_epi] = 0
    lym_mask[pred_result_lym < pred_result_mac] = 0
    lym_mask[pred_result_lym < pred_result_neu] = 0

    mac_mask = np.ones(pred_result_epi.shape)
    mac_mask[pred_result_mac < pred_result_epi] = 0
    mac_mask[pred_result_mac < pred_result_lym] = 0
    mac_mask[pred_result_mac < pred_result_neu] = 0

    neu_mask = np.ones(pred_result_epi.shape)
    neu_mask[pred_result_neu < pred_result_epi] = 0
    neu_mask[pred_result_neu < pred_result_lym] = 0
    neu_mask[pred_result_neu < pred_result_mac] = 0

    pred_result_epi *= epi_mask
    pred_result_lym *= lym_mask
    pred_result_mac *= mac_mask
    pred_result_neu *= neu_mask

    return pred_result_epi, pred_result_lym, pred_result_mac, pred_result_neu


# mask creation
def create_mask(pred_result, size_limit=50):
    bw = otsu_threshold(pred_result)

    no_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bw)
    df = pd.DataFrame(stats, columns=['left', 'top', 'width', 'height', 'area'])
    small = df.index[df['area'] < size_limit].tolist()
    for sm in small:
        labels[labels == sm] = 0
    labels[labels > 0] = 255

    return labels.astype('uint8')


def otsu_threshold(prediction):
    _prediction = np.squeeze(prediction * 255)
    _prediction = _prediction.astype(dtype='uint8')
    _, mask = cv2.threshold(_prediction, 0, 255, cv2.THRESH_OTSU)

    return mask


# -------------------- COLLECTED OUTPUT -----------------------

def collect_output(model, img, stride=128, patch_size=128, padding=0):
    if img.shape[0] < patch_size or img.shape[1] < patch_size:
        old_shape = img.shape[0], img.shape[1]
        shorter = min(img.shape[0], img.shape[1])
        ratio = patch_size / shorter
        dim0 = int(math.ceil(img.shape[0] * ratio))
        dim1 = int(math.ceil(img.shape[1] * ratio))
        img = cv2.resize(img, (dim0, dim1))
    else:
        old_shape = -1

    patch_sums = np.zeros((img.shape[0], img.shape[1]))
    patch_counts = np.zeros((img.shape[0], img.shape[1]))

    patch_pad = patch_size - padding
    stride = stride - 2 * padding

    last_patch_vert = img.shape[0] - patch_size
    last_stride_vert = math.ceil((last_patch_vert) / stride) * stride
    last_patch_horiz = img.shape[1] - patch_size
    last_stride_horiz = math.ceil((last_patch_horiz) / stride) * stride

    i = 0
    while i + patch_size <= img.shape[0]:
        j = 0
        while j + patch_size <= img.shape[1]:

            xs = padding if i != 0 else 0
            xe = patch_pad if i != last_patch_vert else patch_size
            ys = padding if j != 0 else 0
            ye = patch_pad if j != last_patch_horiz else patch_size

            crop = img[i: i + patch_size, j: j + patch_size]
            predicted = model.predict(np.expand_dims(crop, axis=0))

            padded_prediction = np.squeeze(predicted)[xs: xe, ys: ye]

            patch_sums[i + xs: i + xe, j + ys: j + ye] += padded_prediction
            patch_counts[i + xs: i + xe, j + ys: j + ye] += 1

            j = j + stride
            if j == last_stride_horiz: j = last_patch_horiz

        i = i + stride
        if i == last_stride_vert: i = last_patch_vert

    collected_mean = patch_sums / patch_counts
    if old_shape != -1: collected_mean = cv2.resize(collected_mean, old_shape)

    return collected_mean
