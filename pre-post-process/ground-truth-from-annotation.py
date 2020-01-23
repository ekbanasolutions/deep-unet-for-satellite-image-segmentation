#!/usr/bin/env python
# coding: utf-8

# In[86]:


import os
import cv2
import json
import tifffile
import numpy as np
from os import sys
from osgeo import gdal
from matplotlib.pyplot import imshow

all_files = os.listdir('./')
all_json_files = []
for file in all_files:
    if file[-5:]=='.json':
        all_json_files.append(file)
print (all_json_files)


# /home/madhav/Ek/Work/Planet.com/Preprocess/notebooks/20190110_155807_ssc2_u0004_pansharpened_clip.tif_enhanced_x_2550_y_7650.json
# /home/madhav/Ek/Work/Planet.com/Preprocess/notebooks/20190429_184443_ssc9_u0002_pansharpened_clip.tif_enhanced_x_5950_y_850.json
# /home/madhav/Ek/Work/Planet.com/Preprocess/notebooks/20190421_185910_ssc6_u0003_pansharpened_clip.tif_enhanced_x_2550_y_1700.json
# /home/madhav/Ek/Work/Planet.com/Preprocess/notebooks/20190110_155807_ssc2_u0004_pansharpened_clip.tif_enhanced_x_3400_y_6800.json
# /home/madhav/Ek/Work/Planet.com/Preprocess/notebooks/20190421_185910_ssc6_u0003_pansharpened_clip.tif_enhanced_x_1700_y_2550.json
# 

# In[58]:


start_file_name = 99
for img_num, json_file_name in enumerate(all_json_files):

#     json_file = open('/home/madhav/Ek/Work/Planet.com/Preprocess/notebooks/20190429_184443_ssc9_u0002_pansharpened_clip.tif_enhanced_x_5950_y_850.json')
    json_file = open('./' + json_file_name)
    print ("Creating "+ str(start_file_name + img_num)+ ".tif file")
    print ("Processing ", json_file_name)
    data = json.load(json_file)

    labels = ['building','road','tree','crop','water']
    polygons = [[],[],[],[],[]]
    for shape in data['shapes']:
        cl = labels.index(shape['label'])    
        polygons[cl].append(shape['points'])

    import numpy
    from PIL import Image, ImageDraw
    
    mask = np.zeros(shape=(5,1850,1850))
    for i,label in enumerate(labels):
        img = Image.new('L', (1850, 1850), 0)
        for polygon in polygons[i]:  
            flat_list = [item for sublist in polygon for item in sublist]
            if len(flat_list) == 2: #How can there be 2 points in a polygon??
                continue
            ImageDraw.Draw(img).polygon(flat_list, outline=255, fill=255)
        mask[i,:,:] = numpy.array(img)

        imshow(mask[2])
    
    sliced_tif_filename = str(start_file_name + img_num) + '.tif'
    mask = np.uint8(mask)


    # save the polygon list to masked tif below
    error_matrix = np.sum(mask, axis = 0)
    error_img = np.zeros((1850, 1850, 3))
    for i in range(1850):
        for j in range(1850):
            item = error_matrix[i, j]
            if item > 255:
                error_img[i,j] = [0, 0, 255]
            if item == 255:
                error_img[i, j] = [255, 255, 255]
            if item == 0:
                error_img[i, j] = [0, 0, 0]
    
    cv2.imwrite('check_'+json_file_name + '.jpg', error_img)
    tifffile.imsave('/home/madhav/Ek/Work/Planet.com/Preprocess/notebooks/ground_truth/' + sliced_tif_filename, mask)

    # /home/madhav/Ek/Work/Planet.com/Planet_Data_Sliced/tif/20190421_185910_ssc6_u0002_pansharpened_clip.tif_x_850_y_6800.tif
    # the above file is saved as 26

    #TODO: Preprocess / Validate the overlay of different class


# ### This is special code to crop 9 images out of 1850 x 1850 images done by data entry ko bhai haru
# Aditya: /home/madhav/Ek/Work/Planet.com/Data-Entry-Annotation/Given/Groupwise/Lot-1-Data-Entry/Aditya/20190110_155807_ssc2_u0004_pansharpened_clip.tif_enhanced_x_5550_y_3700.jpeg

# In[91]:


aditya1850jpeg = cv2.imread('/home/madhav/Ek/Work/Planet.com/Data-Entry-Annotation/Given/Groupwise/Lot-1-Data-Entry/Aditya/20190110_155807_ssc2_u0004_pansharpened_clip.tif_enhanced_x_5550_y_3700.jpeg')
tiff_path = '/home/madhav/Ek/Work/Planet.com/Preprocess/notebooks/'
tiff_file = '20190110_155807_ssc2_u0004_pansharpened_clip.tif_x_5550_y_3700.tif'


# In[94]:


# #For Aditya's work

# mid_points = [425, 912, 1400]
# img_file = tiff_path + tiff_file
# print (img_file)
# raster = gdal.Open(img_file, gdal.GA_ReadOnly)

# band1 = raster.GetRasterBand(1)
# band2 = raster.GetRasterBand(2)
# band3 = raster.GetRasterBand(3)
# band4 = raster.GetRasterBand(4)

# xBlockSize = 850
# yBlockSize = 850 
# for i, x_mid in enumerate(mid_points):
#     for j, y_mid in enumerate(mid_points):
#         new_cropped_mask = mask[:, x_mid-425:x_mid+425, y_mid-425:y_mid+425]
#         new_cropped_img = aditya1850jpeg[x_mid-425:x_mid+425, y_mid-425:y_mid+425, :]
#         print (new_cropped_img.shape)
#         tifffile.imsave('20190110_155807_ssc2_u0004_pansharpened_clip.tif_enhanced_x_5550_y_3700_' + str(i) + '_' + str(j)+'_GT.tif', new_cropped_mask)
#         cv2.imwrite('20190110_155807_ssc2_u0004_pansharpened_clip.tif_enhanced_x_5550_y_3700_' + str(i) + '_' + str(j)+'.jpeg',new_cropped_img)
#         data1 = band1.ReadAsArray(x_mid-425, y_mid-425, xBlockSize, yBlockSize)
#         data2 = band2.ReadAsArray(x_mid-425, y_mid-425, xBlockSize, yBlockSize)
#         data3 = band3.ReadAsArray(x_mid-425, y_mid-425, xBlockSize, yBlockSize)
#         data4 = band4.ReadAsArray(x_mid-425, y_mid-425, xBlockSize, yBlockSize)
#         temp_for_tiff = np.stack([data3, data2, data1, data4], axis=0)
#         tifffile.imsave('20190110_155807_ssc2_u0004_pansharpened_clip.tif_enhanced_x_5550_y_3700_' + str(i) + '_' + str(j)+'.tif', temp_for_tiff)
#         print ('20190110_155807_ssc2_u0004_pansharpened_clip.tif_enhanced_x_5550_y_3700_' + str(i) + '_' + str(j)+'.tif')
        
    

        


# In[96]:


#For Rubish's work
filename_prefix = '20190421_185910_ssc6_u0002_pansharpened_clip.tif_enhanced_x_1850_y_5550_'

rubish1850jpeg = cv2.imread('/home/madhav/Ek/Work/Planet.com/Data-Entry-Annotation/Given/Groupwise/Lot-1-Data-Entry/Rubish/20190421_185910_ssc6_u0002_pansharpened_clip.tif_enhanced_x_1850_y_5550.jpeg')

tiff_path = '/home/madhav/Ek/Work/Planet.com/Preprocess/notebooks/'
tiff_file = '20190421_185910_ssc6_u0002_pansharpened_clip.tif_x_1850_y_5550.tif'


#why the midpoints??
# conversion of larger image into subseqent smaller image of 850x850 size


mid_points = [425, 912, 1400]
img_file = tiff_path + tiff_file
print (img_file)
raster = gdal.Open(img_file, gdal.GA_ReadOnly)

band1 = raster.GetRasterBand(1)
band2 = raster.GetRasterBand(2)
band3 = raster.GetRasterBand(3)
band4 = raster.GetRasterBand(4)

xBlockSize = 850
yBlockSize = 850 
for i, x_mid in enumerate(mid_points):
    for j, y_mid in enumerate(mid_points):

        
        new_cropped_mask = mask[:, x_mid-425:x_mid+425, y_mid-425:y_mid+425]
        new_cropped_img = rubish1850jpeg[x_mid-425:x_mid+425, y_mid-425:y_mid+425, :]
        print (new_cropped_img.shape)

        
        tifffile.imsave(filename_prefix+ str(i) + '_' + str(j)+'_GT.tif', new_cropped_mask)
        cv2.imwrite(filename_prefix + str(i) + '_' + str(j)+'.jpeg',new_cropped_img)
        data1 = band1.ReadAsArray(x_mid-425, y_mid-425, xBlockSize, yBlockSize)
        data2 = band2.ReadAsArray(x_mid-425, y_mid-425, xBlockSize, yBlockSize)
        data3 = band3.ReadAsArray(x_mid-425, y_mid-425, xBlockSize, yBlockSize)
        data4 = band4.ReadAsArray(x_mid-425, y_mid-425, xBlockSize, yBlockSize)
        temp_for_tiff = np.stack([data3, data2, data1, data4], axis=0)
        tifffile.imsave(filename_prefix + str(i) + '_' + str(j)+'.tif', temp_for_tiff)
        print (filename_prefix + str(i) + '_' + str(j)+'.tif')
        
    

        


# In[ ]:




