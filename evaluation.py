import os
import cv2
import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt

from models import *
from utilities import *
from masks import *


def evaluate_patches(weights_file_path, img_patches_path, mask_patches_path, ignore_rotated=True):

    results = pd.DataFrame(columns=['ji_med', 'dice_med', 'ji', 'dice', 'patch'])

    # load model
    model = base_model(verbose=False)
    model.load_weights(weights_file_path)

    # get names
    names = [f for f in os.listdir(img_patches_path) if ((ignore_rotated and f.count('_') < 3) or not ignore_rotated) and f.endswith('.png')]

    # LOOP
    for idx, name in enumerate(names):
        # load image
        img = cv2.imread(img_patches_path + name)

        # load mask
        mask = cv2.imread(mask_patches_path + name, cv2.IMREAD_GRAYSCALE)

        # pred_array - with rotations
        to_predict = rotate(img)

        # predict
        predicted = model.predict(to_predict, verbose=0)

        # rotate back
        back_rot = rotate_back(predicted)

        # get median of back_rotated
        med = np.median(back_rot, axis=0)

        # calculate mask for first of pred. array (otsu)
        non_med = np.squeeze(predicted[0] * 255)
        non_med = non_med.astype(dtype='uint8')
        _, non_med_mask = cv2.threshold(non_med, 0, 255, cv2.THRESH_OTSU)

        # calculate mask for median of pred. array
        med = np.squeeze(med * 255)
        med = med.astype(dtype='uint8')
        _, med_mask = cv2.threshold(med, 0, 255, cv2.THRESH_OTSU)

        # calculate ji for first pred. array mask
        ji_non_med_mask = metrics.jaccard_score(list(mask.flatten()/255), list(non_med_mask.flatten()/255))
        dice_non_med_mask = metrics.f1_score(list(mask.flatten()/255), list(non_med_mask.flatten()/255))

        # calculate ji for median pred. array mask
        ji_med_mask = metrics.jaccard_score(list(mask.flatten()/255), list(med_mask.flatten()/255))
        dice_med_mask = metrics.f1_score(list(mask.flatten()/255), list(med_mask.flatten()/255))

        # store results in the dictionary
        results.loc[idx] = [ji_med_mask, dice_med_mask, ji_non_med_mask, dice_non_med_mask, name]
    
    return results



def evaluate_patches_colab(weights_file_path, info, image_path, mask_path, dim=(128,128), model_depth=0, c_metrics=False):

    img_names = info.img.unique().tolist()
    images = {}
    masks = {}
    for img_name in img_names:
        images[img_name] = cv2.imread(image_path + img_name + '.tif')
        masks[img_name] = cv2.imread(mask_path + img_name + '.png', cv2.IMREAD_GRAYSCALE)
    
    results = pd.DataFrame(columns=['c_dice', 'c_dice_median', 'patch']) if c_metrics else pd.DataFrame(columns=['ji_med', 'dice_med', 'ji', 'dice', 'patch'])

    # load model
    model = base_model(model_depth=model_depth, verbose=False)
    model.load_weights(weights_file_path)

    # LOOP
    for idx in range(len(info)):
        row = info.loc[idx]
        # load image
        img = images[row.img][row.c_stride : row.c_stride + dim[0] , row.r_stride : row.r_stride + dim[1]]

        # load mask
        mask = masks[row.img][row.c_stride : row.c_stride + dim[0] , row.r_stride : row.r_stride + dim[1]]

        # pred_array - with rotations
        to_predict = rotate(img)

        # predict
        predicted = model.predict(to_predict, verbose=0)

        # rotate back
        back_rot = rotate_back(predicted)

        # get median of back_rotated
        med = np.median(back_rot, axis=0)
        
        # patch name
        name = row.img + '_' + str(row.c_stride) + '_' + str(row.r_stride) + '.png'
        
        if c_metrics:
            c_dice = continuous_dice_coefficient(mask.flatten()/255, np.squeeze(predicted[0]).flatten())
            c_dice_med = continuous_dice_coefficient(mask.flatten()/255, np.squeeze(med).flatten())
            results.loc[idx] = [c_dice, c_dice_med, name]
        else:
            # calculate mask for first of pred. array (otsu)
            non_med = np.squeeze(predicted[0] * 255)
            non_med = non_med.astype(dtype='uint8')
            _, non_med_mask = cv2.threshold(non_med, 0, 255, cv2.THRESH_OTSU)

            # calculate mask for median of pred. array
            med = np.squeeze(med * 255)
            med = med.astype(dtype='uint8')
            _, med_mask = cv2.threshold(med, 0, 255, cv2.THRESH_OTSU)

            # calculate ji for first pred. array mask
            ji_non_med_mask = metrics.jaccard_score(list(mask.flatten()/255), list(non_med_mask.flatten()/255))
            dice_non_med_mask = metrics.f1_score(list(mask.flatten()/255), list(non_med_mask.flatten()/255))

            # calculate ji for median pred. array mask
            ji_med_mask = metrics.jaccard_score(list(mask.flatten()/255), list(med_mask.flatten()/255))
            dice_med_mask = metrics.f1_score(list(mask.flatten()/255), list(med_mask.flatten()/255))

            # store results in the dictionary
            results.loc[idx] = [ji_med_mask, dice_med_mask, ji_non_med_mask, dice_non_med_mask, name]
    
    return results



def inspect_prediction(model, img_name, PATCH_IMG_PATH, PATCH_MASK_PATH, mask_function=otsu_threshold):
    img = cv2.imread(PATCH_IMG_PATH + img_name)
    mask = cv2.imread(PATCH_MASK_PATH + img_name, cv2.IMREAD_GRAYSCALE)

    to_pred = np.zeros((1,*img.shape))
    to_pred[0] = img
    pred = model.predict(to_pred, verbose=0)

    # Plot
    fig=plt.figure(figsize=(30, 10))
    # input
    ax = fig.add_subplot(1, 6, 1)
    ax.title.set_text('Input')
    plt.imshow(img)
    # output
    ax = fig.add_subplot(1, 6, 2)
    ax.title.set_text('Output')
    plt.imshow(cv2.cvtColor(np.squeeze(pred), cv2.COLOR_BGR2RGB))
    # ground truth
    ax = fig.add_subplot(1, 6, 3)
    ax.title.set_text('Ground truth')
    plt.imshow(mask, cmap='gray')
    # our result
    result = mask_function(pred)
    ax = fig.add_subplot(1, 6, 4)
    ax.title.set_text('Our mask')
    plt.imshow(result, cmap='gray')
    # overlay ground truth and our result
    overlay = np.zeros(img.shape, dtype='uint8')
    overlay[:, :, 0] = mask
    overlay[:, :, 1] = result
    ax = fig.add_subplot(1, 6, 5)
    ax.title.set_text('Ground truth vs mask (green)')
    plt.imshow(overlay)
    # overlay our result and input
    contours,_ = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    marked_in_img = cv2.drawContours(img, contours, -1, (0,255,0))
    ax = fig.add_subplot(1, 6, 6)
    ax.title.set_text('Mask contours')
    plt.imshow(marked_in_img)
    plt.show()
    
    
def describe_results(results_file_name, weights_file_path, PATCH_IMG_PATH, PATCH_MASK_PATH):
    
    # load model
    model = base_model(verbose=False)
    model.load_weights(weights_file_path)
    
    # load results
    df = pd.read_csv(results_file_name, index_col=0)
    
    # describe results
    print(df.describe())

    # plot the worst and the best segmentation
    min_ji = df[df.ji_med == df.ji_med.min()]['patch'].item()
    print('The worst: ' + min_ji)
    #inspect_prediction(model, min_ji, PATCH_IMG_PATH, PATCH_MASK_PATH)

    max_ji = df[df.ji_med == df.ji_med.max()]['patch'].item()
    print('The best: ' + max_ji)
   # inspect_prediction(model, max_ji, PATCH_IMG_PATH, PATCH_MASK_PATH)
  