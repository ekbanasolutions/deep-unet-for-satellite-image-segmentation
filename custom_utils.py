import numpy as np
from PIL import Image

band_r, band_g, band_b, band_n = 4, 2, 1, 7 # could be 4, 2, 1, 6 or 4, 2, 1, 7

def get_4bands(img_nband):
    
    '''returns 16 bit, 4 band in the order RGBN, the input should be (width, height, channel)'''
    if (img_nband.shape[2] == 8):
        # multiply by 32 to return true uint16 image (max = 65536)
        newimage = 32*np.stack([img_nband[:,:,4], img_nband[:,:,2], img_nband[:,:,1], img_nband[:,:,7]], axis=-1)
        print ("8 band file given for training, reading only 4 bands with index 4, 2, 1, 7")
    elif (img_nband.shape[2] == 4):
        # the following RGBN order comes from # case: 1001 on Preprocess/slice-planet-tif.ipynb
        newimage = np.stack([img_nband[:,:,0], img_nband[:,:,1], img_nband[:,:,2], img_nband[:,:,3]], axis=-1)
        print ("4 band file given for training, reading rgbn in the index order 0 1 2 3")
    
    return newimage, img_nband.shape[2]

def get_rgb_from_4bands(img_4band):
    newimage = img_4bands[:,:,0:3]
    newimage = np.uint8(newimage/256)
    return newimage

def enhance_rgb(temp):
    new_min = min (temp.flatten())
    new_max = max (temp.flatten())
    k1 = (255-new_min)/(new_max-new_min)
    k2 = (new_max-255)*new_min/(new_max-new_min)
    temp = np.uint8(temp*k1 + k2)
    return temp

def save_to_verify(img, filename, original_bands):
    '''save 3 band rgb for verifiation, enhance planet rgb before saving'''
    newimage = get_rgb(img)
    if original_bands == 4: #for planet dataset. Enhancement not needed for kagle dataset
        newimage = enhance_rgb(newimage)
    print ("4 band file saving as RGB jpeg for verification.")
    jpeg_image = Image.fromarray(newimage)
    jpeg_image.save('data/verify/' + filename + '.jpeg')
    return newimage

