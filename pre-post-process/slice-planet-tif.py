#!/usr/bin/env python
# coding: utf-8

# # Do Preliminary tasks to large tiff files
# This includes:
# * clipping the file to smaller tifs
# * calculating the information of the latlong of corners of smaller tif
# * storing the information thus calculate to json
# 

# ## Import Libs

# In[1]:


import numpy as np
from os import sys
import tifffile
from osgeo import gdal
from PIL import Image, ImageEnhance
import matplotlib.pyplot
import json


# ## Define functions and (large) tif path

# In[2]:

planet_dir = "/home/ekbana/computer_vision/satellite-image/Planet.com/" 

def enhance_img(temp_for_im):
    ###enhancement begins
    new_min = min (temp_for_im.flatten())
    new_max = max (temp_for_im.flatten())
    k1 = (255-new_min)/(new_max-new_min)
    k2 = (new_max-255)*new_min/(new_max-new_min)
    temp_for_im = np.uint8(temp_for_im*k1 + k2)                       
    ###enhancement ends
    return temp_for_im

tiff_path = planet_dir + 'Planet-Data/'
tiff_file = '20190421_185910_ssc6_u0002_pansharpened_clip.tif'
img_path = tiff_path + tiff_file
raster = gdal.Open(img_path, gdal.GA_ReadOnly)

# GDAL affine transform parameters
# According to gdal documentation xoff/yoff are image left corner,
# a/e are pixel width/height and b/d is rotation and is zero if image is north up.
x_origin, a, b, y_origin, d, e = raster.GetGeoTransform()

cols = raster.RasterXSize
rows = raster.RasterYSize
bands= raster.RasterCount

print ("The size of the geotiff file is", cols, rows, bands)
print ("x_origin, a, b, y_origin, d, e = ", x_origin, a, b, y_origin, d, e)

# GDAL affine transform parameters
# According to gdal documentation xoff/yoff are image left corner,
# a/e are pixel width/height and b/d is rotation and is zero if image is north up.

def pixel2coordinates(x, y):
    """Returns global coordinates from pixel x, y coords"""
    #TODO --> get x_offset, y_offset for each individual slice
    xp = a * x + b * y + x_origin + x_offset
    yp = d * x + e * y + y_origin + y_offset
    return(xp, yp)


# In[3]:


band1 = raster.GetRasterBand(1)
band2 = raster.GetRasterBand(2)
band3 = raster.GetRasterBand(3)
band4 = raster.GetRasterBand(4)

blockSize = band1.GetBlockSize()
# print (blockSize) #is 256, 256 for the planet data.

# xBlockSize = blockSize[0]
# yBlockSize = blockSize[1]

xBlockSize = 850
yBlockSize = 850 

WRITE_TIF = 1 # Make this 1 to write the TIF FiLE
WRITE_JPEG = 1 # Make this 1 to write the TIF FiLE
SHOW_IMG = 0 #Make this 0 to show the image on notebook


json_data = {
    "original_file":tiff_file,
    "xBlockSize": xBlockSize,
    "yBlockSize": yBlockSize,
    "geotransform-params":{
        "x-origin":x_origin,
        "y-origin":y_origin,
        "pixel-width":a,
        "b":b,
        "d":d,
        "pixel-height":e
    }
}

for rangex in range (int(cols/xBlockSize)):
    for rangey in range (int(rows/yBlockSize)):
        # print ('Block Readed')
        data1 = band1.ReadAsArray(rangex*yBlockSize, rangey*yBlockSize, xBlockSize, yBlockSize)
        # print (data1)
        if True or (rangex*xBlockSize == 850 and rangey*yBlockSize == 5100): #True or (rangex*xBlockSize == 850 and rangey*yBlockSize == 5100):
            if sum(data1.flatten()) > 0:

                # print (rangex*xBlockSize, rangey*yBlockSize)
                x_offset = rangex*xBlockSize
                y_offset = rangey*yBlockSize
                
                data2 = band2.ReadAsArray(rangex*xBlockSize, rangey*yBlockSize, xBlockSize, yBlockSize)
                data3 = band3.ReadAsArray(rangex*xBlockSize, rangey*yBlockSize, xBlockSize, yBlockSize)
                data4 = band4.ReadAsArray(rangex*xBlockSize, rangey*yBlockSize, xBlockSize, yBlockSize)
                
                # temp_for_tiff = np.stack([data1, data2, data3, data4], axis=0) #visualizes little bit incorrectly                
                # below is the right order for PIL Image [data3, data2, data1, data4]
                # visualizes nicely!! (in document viewer as well)
                # saving as (4, xBlockSize, yBlockSize) as all the kaggle data are in (8,width,height)
                # case: 1001
                temp_for_tiff = np.stack([data3, data2, data1, data4], axis=0)
                
                # print (temp_for_tiff.shape)
                sliced_tif_filename = tiff_file + '_x_' + str(rangex*xBlockSize) + '_y_'+ str(rangey*yBlockSize) + '.tif'
                json_data[sliced_tif_filename] = {}
                if WRITE_TIF == 1:
                    tifffile.imsave(planet_dir + 'Planet_Data_Sliced/tif/' + sliced_tif_filename, temp_for_tiff)
                    print ('TIF file saved succesfully. Filename: ', sliced_tif_filename)                           
                
                #------------------------ saving jpeg from saved tif data -----------------------
                
                # following is a redundant thing to do
                # and could be done directly from gdal data1, data2, data3, data4 instead as done in the code block above
                # but I am doing this to maintain integrity and ensure tiffs are well written
        
                raw_im_data = tifffile.imread(planet_dir + 'Planet_Data_Sliced/tif/' + sliced_tif_filename)
                # print (raw_im_data.shape, max(raw_im_data.flatten())) (4, xBlockSize, yBlockSize), 65536 ==> 65536 = 16 bit
                
                raw_rgb_im = raw_im_data[0:3,:,:]
                raw_rgb_im = raw_rgb_im.transpose([1,2,0])                 # to transform 3, xBlockSize, yBlockSize to xBlockSize, yBlockSize, 3
                enhanced_im2 = np.uint8(enhance_img(raw_rgb_im/256))       # to transform from (65536)16bit to (256)8 bit
                
                if SHOW_IMG == 1:
                    tifffile.imshow(enhanced_im2)
                    
                image = Image.fromarray(enhanced_im2)
                sliced_jpeg_filename = tiff_file + '_enhanced_x_' + str(rangex*xBlockSize) + '_y_'+ str(rangey*yBlockSize) + '.jpeg' 
                if WRITE_JPEG== 1:
                    image.save(planet_dir + 'Planet_Data_Sliced/jpg/' + sliced_jpeg_filename)
                    print ("JPEG file saved successfully from second method. Filename: ",sliced_jpeg_filename)
                                    
                json_info_file = tiff_file + "_info.json"
                json_file_path = tiff_path + json_info_file                    
                print ("working for ", sliced_tif_filename)
                
                # TODO correct conversion
                top, left = pixel2coordinates(rangex*xBlockSize, rangey*yBlockSize)
                bottom, right = pixel2coordinates(xBlockSize*(rangex+1)-1, yBlockSize*(rangey+1)-1)
                # print ("The coordinates of the corner for image at {} {} are {} {} {} {}".format(rangex*xBlockSize, rangey*yBlockSize, top, left, bottom, right))
                json_data[sliced_tif_filename]['top-left'] = [top, left]
                json_data[sliced_tif_filename]['bottom-right'] = [bottom, right]
                
                # pixel offset below (of the top left corner) is wrt original large tif's top left corner
                json_data[sliced_tif_filename]["pixel-offset"] = {    
                    "x-off":x_offset,
                    "y-off":y_offset
                }


with open(json_file_path, 'w') as f:
    json.dump(json_data, f, indent=4)

# print (json.dumps(json_data, indent=4))

