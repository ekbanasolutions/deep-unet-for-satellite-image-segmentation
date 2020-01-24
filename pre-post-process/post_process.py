#!/usr/bin/env python
# coding: utf-8

# # Postprocess the polygons to GeoData
# This includes:
# * reading json data of predicted polygons
# * reading json data of meta of the sliced tiffs
# * converting boundary / outline of the polygons to lat/long from pixels
# 

# ## Import Libs

# In[3]:


import os
import numpy as np
from os import sys
import json
import utm

from shapely.geometry import mapping, Polygon
import fiona

import tifffile as tiff
from map_to_json import postprocess_masks, mask_array_to_poly_json
from predict import picture_from_mask

result_path = "../../Planet.com/Planet_Data_Sliced/tif/result/"
all_result_tifs = os.listdir(result_path)
all_map_tifs = [file for file in all_result_tifs if file[-8:] == "_map.tif"]
all_result_tifs = [file for file in all_result_tifs if file[-11:] == "_result.tif"]
total_files_count = len(all_result_tifs)

for current_file_count, (result_tif, map_tif) in enumerate(zip(all_result_tifs, all_map_tifs)):
    mymat = tiff.imread(result_path + result_tif)
    mymat = mymat / 255 # (convert grayscale value uint8 to probability matrix) but it might not be necessary
    mymap = tiff.imread(result_path + map_tif)
    print ("...creating binary mask of the result from probability matrix of result")
    bin_mask_array = postprocess_masks(mymat, mymat.shape)
    print ("...saving binary mask to polygon of each class")
    mask_array_to_poly_json(bin_mask_array, result_path, os.path.split(result_tif)[1], reqd_class_label=['Trees', 'Crops', 'Water'])
    print ("Written Polygons into json for {}".format(result_tif))
    print ("Completed {} out of {}".format(current_file_count+1, total_files_count))

# In[4]:


x1, y1 = 358011.19999999995, 3038709.5999999996
x2,y2 = utm.to_latlon(x1,y1, 17, 'N')
print (x2, y2)


# ## Define functions and path

# In[5]:


result_path = '/home/ekbana/computer_vision/satellite-image/Planet.com/Planet_Data_Sliced/tif/result/'
post_proc_temp_path = result_path + 'Post-Process-Temp/'
result_filenames = os.listdir(result_path)
result_filenames = [file for file in result_filenames if file[-10:] == 'polys.json']

meta_path = '/home/ekbana/computer_vision/satellite-image/Planet.com/Planet-Data/'

all_meta_files = os.listdir(meta_path)
all_meta_files = [meta_filename for meta_filename in all_meta_files if meta_filename[-9:] == 'info.json']

def get_georef(this_slice, meta_data):
    return meta_data[this_slice]


def pixel2latlong(x, y, transform_params, tif_slice_georef):
    """Returns global coordinates from pixel x, y coords"""
    # GDAL affine transform parameters
    # According to gdal documentation xoff/yoff are image left corner,
    # a/e are pixel width/height and b/d is rotation and is zero if image is north up. 
    
    pixel_offset_x = tif_slice_georef['pixel-offset']['x-off']
    pixel_offset_y = tif_slice_georef['pixel-offset']['y-off']
    
    a = transform_params['pixel-width']
    e = transform_params['pixel-height']
    b = transform_params['b']
    d = transform_params['d']
    x_origin = transform_params['x-origin']
    y_origin = transform_params['y-origin']
    
    
    
    #print (pixel_offset_x, pixel_offset_y)

    # print ("--")
    # print ("a = ", a)
    # print (x)
    # print (b)
    # print (y)
    # print (x_origin)
    # print (y_origin)
    # print (pixel_offset_x)
    
    # xp = a * x + b * y + x_origin + pixel_offset_x*a
    # yp = d * x + e * y + y_origin + pixel_offset_y*e

    xp = a * (pixel_offset_x + x) + b * (pixel_offset_y + y) + x_origin
    yp = d * (pixel_offset_x + x) + e * (pixel_offset_y + y) + y_origin

    
    lat, long = utm.to_latlon(xp, yp, 17, 'N')
    return(long, lat)

def pixeldata_to_latlongdata(poly_data_pixel, tif_slice_georef):
    class_label = {
        0: "Buildings",
        1: "Roads & Tracks",
        2: "Trees",
        3: "Crops",
        4: "Water"
    }
    for cl in range(5): #do only for trees, otherwise do range(5)
        cl_data = poly_data_pixel['details'][class_label[cl]]
        print (len(cl_data), "data found in class_label: ", class_label[cl])
        new_cl_data = {}
        for key, polygon in cl_data.items():
            new_cl_data[str(key)] = []
            for j in range(0,len(polygon),2):
                # print ("Processing vertex " , j , j+1)
                # polygon[j],polygon[j+1] = pixel2latlong(polygon[j],polygon[j+1],transform_params)
                pol_xy = tuple(pixel2latlong(polygon[j],polygon[j+1],transform_params, tif_slice_georef))
                new_cl_data[str(key)].append(pol_xy)
        poly_data_pixel['details'][class_label[cl]] = new_cl_data
    return poly_data_pixel

def latlongdata_to_shapefile(poly_latlong_data, filename, shapefile_handler, original_tif):
    print ("Writing multipolygon shp file.")
    
    for class_label, cl_data in poly_latlong_data['details'].items():
        if class_label == 'Water':
            
            # cl_data = poly_latlong_data['details'][class_label]
            for key, polygon  in cl_data.items():
                try:
                    poly = Polygon (polygon)        
                except Exception as e:
                    print ("ERROR trying to create polygon for keyid: ", key)
                    # raise e 
                
                shapefile_handler.write({
                    'geometry': mapping(poly),
                    'properties': {
                        'class': class_label,
                        'id':int(key),
                        'original_tif':original_tif
                    }
                })

            print ("Finished writing for ", class_label, "for this slice")

# for meta_filename in all_meta_files:




schema = {
        'geometry': 'Polygon',
        'properties': {'class': 'str', 'id': 'int', 'original_tif': 'str'},
    }

previous_original_tif = None
result_filenames = sorted(result_filenames)
c = fiona.open("/home/ekbana/computer_vision/satellite-image/Planet.com/Planet_Data_Sliced/tif/result/Postprocess-Result/result_all.shp", 'w', 'ESRI Shapefile', schema)
for result_filename in result_filenames:
    print ("...post processing for {}".format(result_filename))
    with open(result_path + result_filename) as json_fp:
        poly_data_pixel = json.loads(json_fp.read())
        this_slice = poly_data_pixel['tif-slice-filename']
        original_tif = poly_data_pixel['tif-slice-filename'].split(".")[0]+'.tif'
        if original_tif != previous_original_tif:
            previous_original_tif = original_tif
            # c = fiona.open("/home/ekbana/computer_vision/satellite-image/Planet.com/Planet_Data_Sliced/tif/result/Postprocess-Result/trees_" + original_tif + "_all.shp", 'w', 'ESRI Shapefile', schema)
        poly_data_pixel['original-tif'] = original_tif
        meta_filename = original_tif + "_info.json"

        with open(meta_path + meta_filename) as meta_fp:
            meta_data = json.loads(meta_fp.read())

            transform_params = meta_data['geotransform-params']
            # geo_ref = meta_data[this_slice]

        #buildings = poly_data['details']['class0']


        geo_ref = get_georef(this_slice, meta_data)
        poly_latlong_data = pixeldata_to_latlongdata(poly_data_pixel, tif_slice_georef = geo_ref)

        with open(post_proc_temp_path + result_filename + "latlong.json", "w") as json_fp:
            json.dump(poly_latlong_data, json_fp, indent=4)
            latlongdata_to_shapefile(poly_latlong_data, result_filename, c, original_tif)
