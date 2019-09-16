import os
os.environ["CUDA_VISIBLE_DEVICES"]="1" # second gpu

import math
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
from os import sys
import itertools

from PIL import Image, ImageDraw

from train_unet import weights_path, get_model, normalize, PATCH_SZ, N_CLASSES 
from custom_utils import get_4bands

def predict(x, model, patch_sz=160, n_classes=5):
    img_height = x.shape[0]
    img_width = x.shape[1]
    n_channels = x.shape[2]
    # make extended img so that it contains integer number of patches
    npatches_vertical = math.ceil(img_height / patch_sz)
    npatches_horizontal = math.ceil(img_width / patch_sz)
    extended_height = patch_sz * npatches_vertical
    extended_width = patch_sz * npatches_horizontal
    ext_x = np.zeros(shape=(extended_height, extended_width, n_channels), dtype=np.float32)
    # fill extended image with mirrors:
    ext_x[:img_height, :img_width, :] = x
    for i in range(img_height, extended_height):
        ext_x[i, :, :] = ext_x[2 * img_height - i - 1, :, :]
    for j in range(img_width, extended_width):
        ext_x[:, j, :] = ext_x[:, 2 * img_width - j - 1, :]

    # now we assemble all patches in one array
    patches_list = []
    for i in range(0, npatches_vertical):
        for j in range(0, npatches_horizontal):
            x0, x1 = i * patch_sz, (i + 1) * patch_sz
            y0, y1 = j * patch_sz, (j + 1) * patch_sz
            patches_list.append(ext_x[x0:x1, y0:y1, :])
    # model.predict() needs numpy array rather than a list
    patches_array = np.asarray(patches_list)
    # predictions:
    patches_predict = model.predict(patches_array, batch_size=4)
    prediction = np.zeros(shape=(extended_height, extended_width, n_classes), dtype=np.float32)
    for k in range(patches_predict.shape[0]):
        i = k // npatches_horizontal
        j = k % npatches_vertical
        x0, x1 = i * patch_sz, (i + 1) * patch_sz
        y0, y1 = j * patch_sz, (j + 1) * patch_sz
        prediction[x0:x1, y0:y1, :] = patches_predict[k, :, :, :]
    return prediction[:img_height, :img_width, :]


def picture_from_mask(mask, threshold=0):
    colors = {
        0: [150, 150, 150],  # Buildings
        1: [223, 194, 125],  # Roads & Tracks
        2: [27, 120, 55],    # Trees
        3: [166, 219, 160],  # Crops
        4: [116, 173, 209]   # Water
    }

    # original z-order = 3,4,0,1,2
    z_order = {
        1: 3,
        2: 2,
        3: 4,
        4: 0,
        5: 1 
    }
    pict = 255*np.ones(shape=(3, mask.shape[1], mask.shape[2]), dtype=np.uint8)

    for i in range(1, 6):
        cl = z_order[i]
        for ch in range(3):
            pict[ch,:,:][mask[cl,:,:] > threshold] = colors[cl][ch]
    return pict

if __name__ == '__main__':
    model = get_model()
    if len(sys.argv) == 3:
        weights_path = sys.argv[2]
    model.load_weights(weights_path)
    if len(sys.argv)<2:
        print("Enter file path")
    test_file = sys.argv[1]
    test_id = sys.argv[1]
    #test_id = '23'
    class_id = 4 
    # img = take_4bands(normalize(tiff.imread('data/mband/{}.tif'.format(test_id)).transpose([1,2,0])))   # make channels last
    img = tiff.imread(test_file).transpose([1,2,0])
    # print (img)
    img, _ = get_4bands(normalize(tiff.imread(test_file).transpose([1,2,0])))   # make channels last
    band_list = [0, 1, 2, 3] # to permute all the bands
    perm_list = list(itertools.permutations(band_list))
    print ("The shape of image for prediction is: ", img.shape)
    res_shape = tuple([8]+list(img.shape[0:2])+[N_CLASSES]) # [8] here comes from the 8 augmented images
    results = np.zeros(shape=res_shape)
    print ("Results Shape = ", results.shape)
    for i in range(8):
        if i == 0:  # reverse first dimension
            mymat = predict(img[::-1,:,:], model, patch_sz=PATCH_SZ, n_classes=N_CLASSES).transpose([2,0,1])
            print(mymat[class_id][0][0], mymat[3][12][13])
            print("Case 1",img.shape, mymat.shape)
        elif i == 1:    # reverse second dimension
            temp = predict(img[:,::-1,:], model, patch_sz=PATCH_SZ, n_classes=N_CLASSES).transpose([2,0,1])
            print(temp[class_id][0][0], temp[3][12][13])
            print("Case 2", temp.shape, mymat.shape)
            mymat = np.mean( np.array([ temp[:,::-1,:], mymat ]), axis=0 )
        elif i == 2:    # transpose(interchange) first and second dimensions
            temp = predict(img.transpose([1,0,2]), model, patch_sz=PATCH_SZ, n_classes=N_CLASSES).transpose([2,0,1])
            print(temp[class_id][0][0], temp[3][12][13])
            print("Case 3", temp.shape, mymat.shape)
            mymat = np.mean( np.array([ temp.transpose(0,2,1), mymat ]), axis=0 )
        elif i == 3:
            temp = predict(np.rot90(img, 1), model, patch_sz=PATCH_SZ, n_classes=N_CLASSES)
            print(temp.transpose([2,0,1])[class_id][0][0], temp.transpose([2,0,1])[3][12][13])
            print("Case 4", temp.shape, mymat.shape)
            mymat = np.mean( np.array([ np.rot90(temp, -1).transpose([2,0,1]), mymat ]), axis=0 )
        elif i == 4:
            temp = predict(np.rot90(img,2), model, patch_sz=PATCH_SZ, n_classes=N_CLASSES)
            print(temp.transpose([2,0,1])[class_id][0][0], temp.transpose([2,0,1])[3][12][13])
            print("Case 5", temp.shape, mymat.shape)
            mymat = np.mean( np.array([ np.rot90(temp,-2).transpose([2,0,1]), mymat ]), axis=0 )
        elif i == 5:
            temp = predict(np.rot90(img,3), model, patch_sz=PATCH_SZ, n_classes=N_CLASSES)
            print(temp.transpose([2,0,1])[class_id][0][0], temp.transpose([2,0,1])[3][12][13])
            print("Case 6", temp.shape, mymat.shape)
            mymat = np.mean( np.array([ np.rot90(temp, -3).transpose(2,0,1), mymat ]), axis=0 )
        elif i ==6:
            #added by me 
            temp = predict(img, model, patch_sz=PATCH_SZ, n_classes=N_CLASSES).transpose([2,0,1])
            print ("Case 7", temp.shape, mymat.shape)
        else:
            temp = predict(img, model, patch_sz=PATCH_SZ, n_classes=N_CLASSES).transpose([2,0,1])
            print(temp[class_id][0][0], temp[3][12][13])
            print("Case 8", temp.shape, mymat.shape)
            mymat = np.mean( np.array([ temp, mymat ]), axis=0 )

    #print(mymat[class_id][0][0], mymat[3][12][13])
    map = picture_from_mask(mymat, 0.3)
        #mask = predict(img, model, patch_sz=PATCH_SZ, n_classes=N_CLASSES).transpose([2,0,1])  # make channels first
    #map = picture_from_mask(mask, 0.5)

    #tiff.imsave('result.tif', (255*mask).astype('uint8'))

    tiff.imsave(test_file + '_result.tif', (255*mymat).astype('uint8'))
    tiff.imsave(test_file + '_map.tif', map)
