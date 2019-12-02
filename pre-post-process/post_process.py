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
    for cl in range(2,3): #do only for trees, otherwise do range(5)
        cl_data = poly_data_pixel['details'][class_label[cl]]
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

def latlongdata_to_shapefile(poly_latlong_data, filename, shapefile_handler):
    print ("Writing multipolygon shp file.")
    
    for class_label, cl_data in poly_latlong_data['details'].items():
        if class_label == 'Trees':
            
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
                        'id':int(key)
                    }
                })

            print ("Finished writing for ", class_label, "for this slice")

# for meta_filename in all_meta_files:




schema = {
        'geometry': 'Polygon',
        'properties': {'class': 'str', 'id':'int'},
    }

previous_original_tif = None
result_filenames = sorted(result_filenames)
c = fiona.open("/home/ekbana/computer_vision/satellite-image/Planet.com/Planet_Data_Sliced/tif/result/Postprocess-Result/trees_all.shp", 'w', 'ESRI Shapefile', schema)
for result_filename in result_filenames:
    print ("...post processing for {}".format(result_filename))
    with open(result_path + result_filename) as json_fp:
        poly_data_pixel = json.loads(json_fp.read())
        this_slice = poly_data_pixel['tif-slice-filename']
        original_tif = poly_data_pixel['tif-slice-filename'].split(".")[0]+'.tif'
        if original_tif != previous_original_tif:
            previous_original_tif = original_tif
            # c = fiona.open("/home/ekbana/computer_vision/satellite-image/Planet.com/Planet_Data_Sliced/tif/result/Postprocess-Result/trees_" + original_tif + ".shp", 'w', 'ESRI Shapefile', schema)
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
            latlongdata_to_shapefile(poly_latlong_data, result_filename, c)
