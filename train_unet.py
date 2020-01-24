import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1" # second gpu

from unet_model import *
from gen_patches import *

import os.path
from os import sys
import numpy as np
import tifffile as tiff
from keras.callbacks import CSVLogger
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from custom_utils import save_to_verify, get_4bands

def normalize(img):
    min = img.min()
    max = img.max()
    x = 2.0 * (img - min) / (max - min) - 1.0
    return x

N_BANDS = 4 
N_CLASSES = 5  # buildings, roads, trees, crops and water
N_EPOCHS = 25 #150 #150 is original value
CLASS_WEIGHTS = [0.2, 0.3, 0.1, 0.1, 0.3] #original
CLASS_WEIGHTS = [0.2, 0.3, 0.1, 0.1, 0.3] #w28

PATCH_SZ = 160  # was originally 160 # should divide by 16
BATCH_SIZE = 50  #150 #150 is original value #runs well on 20 but.. 
UPCONV = True
TRAIN_SZ = 4000  # train size
VAL_SZ = 1000

def get_model():
    return unet_model(N_CLASSES, PATCH_SZ, n_channels=N_BANDS, upconv=UPCONV, class_weights=CLASS_WEIGHTS)

weights_path = 'weights'
if not os.path.exists(weights_path):
    os.makedirs(weights_path)

trainIds = [str(i).zfill(2) for i in range(1, 50)]  # all availiable ids: from "1" to "59" #w28

if __name__ == '__main__':
    if len(sys.argv)<2:
        print("Enter weight path (eg. w37.hdf5)")
        sys.exit()

    weights_path += '/' + sys.argv[1]

    X_DICT_TRAIN = dict()
    Y_DICT_TRAIN = dict()
    X_DICT_VALIDATION = dict()
    Y_DICT_VALIDATION = dict()

    print('Reading images')
    for img_id in trainIds:
        tif_img = tiff.imread('./data/mband/{}.tif'.format(img_id))
        print ("Shape of input tif file",tif_img.shape)
        tif_img = tif_img.transpose([1, 2, 0])
        img_4bands, original_bands = get_4bands(tif_img)
        # save_to_verify(img_4bands, str(img_id), original_bands)
        
        img_4bands = normalize(img_4bands)

        mask = tiff.imread('./data/gt_mband/{}.tif'.format(img_id)).transpose([1, 2, 0]) / 255
        train_xsz = int(3/4 * img_4bands.shape[0])  # use 75% of image as train and 25% for validation
        X_DICT_TRAIN[img_id] = img_4bands[:train_xsz, :, :]
        Y_DICT_TRAIN[img_id] = mask[:train_xsz, :, :]
        X_DICT_VALIDATION[img_id] = img_4bands[train_xsz:, :, :]
        Y_DICT_VALIDATION[img_id] = mask[train_xsz:, :, :]
        print(img_id + ' read')
    print('Images were read')

    def train_net():
        print("start train net")
        x_train, y_train = get_patches(X_DICT_TRAIN, Y_DICT_TRAIN, n_patches=TRAIN_SZ, sz=PATCH_SZ)
        x_val, y_val = get_patches(X_DICT_VALIDATION, Y_DICT_VALIDATION, n_patches=VAL_SZ, sz=PATCH_SZ)
        model = get_model()
        if os.path.isfile(weights_path):
            model.load_weights(weights_path)
        #model_checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', save_weights_only=True, save_best_only=True)
        #early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')
        #reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5, min_lr=0.00001)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.01, patience=5, min_lr=0.00001)
        model_checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', save_best_only=True)
        csv_logger = CSVLogger('log_unet.csv', append=True, separator=';')
        tensorboard = TensorBoard(log_dir='./tensorboard_unet/', write_graph=True, write_images=True)
        model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=N_EPOCHS,
                  verbose=2, shuffle=True,
                  callbacks=[model_checkpoint, csv_logger, tensorboard],
                  validation_data=(x_val, y_val))
        return model

    train_net()


