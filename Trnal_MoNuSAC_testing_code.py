# This is the test code to predict different cell classes
# Author Trnal (for MoNuSAC challenge 2020)

import cv2
import numpy as np
from glob import glob
import scipy.io
import openslide

import os
import sys

sys.path.append(os.path.dirname(os.getcwd()))

from custom_functions import *

#
# ------------- IMPORTANT -----------------
# these paths need to be set before running the program
# save_as_mat - if false results will be stored as .tif images
#
data_path = '/Users/macbook/Desktop/SKOLA/DP/Data/MoNuSAC_images_and_annotations/data/'
dest_path = os.path.dirname(os.getcwd()) + '/monusac_results/'
MODELS_PATH = os.path.dirname(os.getcwd()) + '/models/'
save_as_mat = True

create_folder(dest_path)

model_epi = base_model(verbose=False)
model_epi.load_weights(MODELS_PATH + 'MNS_epitel.h5')

model_lym = base_model(verbose=False)
model_lym.load_weights(MODELS_PATH + 'MNS_lymph.h5')

model_mac = base_model(verbose=False)
model_mac.load_weights(MODELS_PATH + 'MNS_macro.h5')

model_neu = base_model(verbose=False)
model_neu.load_weights(MODELS_PATH + 'MNS_neutro.h5')

patients = [x[0] for x in os.walk(data_path)]
if data_path in patients: patients.remove(data_path)

for patient in patients:

    patient_name = patient[len(data_path) + 1:]  # Patient name
    print(patient_name)

    # create patient directory
    create_folder(dest_path + patient_name)

    # for each image in patient
    sub_images = glob(patient + '/*.svs')
    for sub_image in sub_images:

        # convert to tif if needed
        if not os.path.isfile(sub_image[:-4] + '.tif'):
            img = openslide.OpenSlide(sub_image)
            cv2.imwrite(sub_image[:-4] + '.tif', np.array(img.read_region((0, 0), 0, img.level_dimensions[0])))

        sub_image_name = sub_image[len(patient) + 1:]
        dest_subimage = dest_path + patient_name + '/' + sub_image_name[:-4] + '/'

        # load image
        img = cv2.imread(sub_image[:-4] + '.tif')

        # create subimage directory and class directories in it
        create_folder(dest_subimage)

        create_folder(dest_subimage + 'Epithelial')
        create_folder(dest_subimage + 'Lymphocyte')
        create_folder(dest_subimage + 'Macrophage')
        create_folder(dest_subimage + 'Neutrophil')

        # get predictions
        epi_p, lym_p, mac_p, neu_p = predict_all_classes(img, model_epi, model_lym, model_mac, model_neu)

        # create masks from predictions
        epi = create_mask(epi_p, size_limit=50)
        lym = create_mask(lym_p, size_limit=50)
        mac = create_mask(mac_p, size_limit=50)
        neu = create_mask(neu_p, size_limit=50)

        # save results
        if save_as_mat:
            scipy.io.savemat(dest_subimage + 'Epithelial/' + sub_image_name[:-4] + '.mat', {epi.dtype.name: epi})
            scipy.io.savemat(dest_subimage + 'Lymphocyte/' + sub_image_name[:-4] + '.mat', {lym.dtype.name: lym})
            scipy.io.savemat(dest_subimage + 'Macrophage/' + sub_image_name[:-4] + '.mat', {mac.dtype.name: mac})
            scipy.io.savemat(dest_subimage + 'Neutrophil/' + sub_image_name[:-4] + '.mat', {epi.dtype.name: neu})
        else:
            cv2.imwrite(dest_subimage + 'Epithelial/' + sub_image_name[:-4] + '.tif', epi)
            cv2.imwrite(dest_subimage + 'Lymphocyte/' + sub_image_name[:-4] + '.tif', lym)
            cv2.imwrite(dest_subimage + 'Macrophage/' + sub_image_name[:-4] + '.tif', mac)
            cv2.imwrite(dest_subimage + 'Neutrophil/' + sub_image_name[:-4] + '.tif', neu)

