import json
import numpy as np
from skimage import measure
from PIL import Image, ImageDraw

def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour


def postprocess_masks(result, image, min_nuc_size=10):

    """Clean overlaps between bounding boxes, fill small holes, smooth boundaries"""
    """Convert probability for each class into (one hot encoded) mask""" 

    height, width = image.shape[:2]
    n_classes = result.shape[0]
    sth = np.zeros(result.shape)

    for row in range(height):
        for col in range(width):
            for cl in range(n_classes):
                temp_val = result[cl,row,col] 
                if result[cl,row,col] == max(result[:,row,col]):
                    result[:,row,col] = np.zeros(n_classes)
                    if temp_val > 0.5:
                        result[cl, row, col] = 1

    return result

def binary_mask_to_polygon(binary_mask, tolerance=0):
    """Converts a binary mask to COCO polygon representation
    Args:
        binary_mask: a 2D binary numpy array where '1's represent the object
        tolerance: Maximum distance from original points of polygon to approximated
            polygonal chain. If tolerance is 0, the original coordinate array is returned.
    """
    polygons = []
    # pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
    contours = measure.find_contours(padded_binary_mask, 0.5)
    contours = np.subtract(contours, 1)
    for contour in contours:
        contour = close_contour(contour)
        contour = measure.approximate_polygon(contour, tolerance)
        if len(contour) < 3:
            continue
        contour = np.flip(contour, axis=1)
        #print(contour)
        segmentation = contour.ravel().tolist()
        # after padding and subtracting 1 we may get -0.5 points in our segmentation
        segmentation = [0 if i < 0 else i for i in segmentation]
        polygons.append(segmentation)

    return polygons


def generate_json(result,image_path,classes):
    j_data =    {
                "flags": {},
                "shapes": [
                            ],
                "lineColor": [
                                 0,
                                 0,
                                  255,
                                2
                             ],
                "fillColor": [
                          255,
                      0,
                         0,
                             2
                            ]
                                }




    null =None
    
    if (result !="None"):
        masks = result['masks']
        for i in  range(masks.shape[-1]):

            try:
                class_ids = str(result["class_ids"][i])

                if class_ids== '1':
                    class_name = classes[0]
                  
                elif class_ids =='2':
                    
                    class_name = classes[1]
                elif class_ids =='3':
                    class_name = classes[2]

            except:
                class_name = None
            
            #print(masks.shape)
            mask = masks[:,:, i ]
            mask  = mask *255
            mask = mask.astype(np.uint8)

            poly = binary_mask_to_polygon(mask)

            print("length of polygon",len(poly))
            print(poly)
            merged_poly = []

            for l in poly:
                merged_poly += l


            poly= np.asarray(merged_poly)

  
            poly =poly.reshape((-1,2))
                        

            poly = np.asarray(poly)
            
            new_poly  =  reduce_polygon_points(poly)

            j_data["shapes"].append({"label":class_name,"line_color":null,"fill_color":null,"points":new_poly})
        
        j_data["imageData"] = null

       # print("imdata",imdata)
        path, filename_image = os.path.split(image_path)
        filename, file_extension = os.path.splitext(filename_image)
        j_data["imagePath"] = filename_image
        with open('json_files/%s.json'%filename, 'w') as outfile:
            json.dump(j_data, outfile, indent=2)

def mask_array_to_poly_json(bin_mask_array, result_path="./result/",tif_filename=None,reqd_class_label=['Trees', 'Crops', 'Water', "Roads & Tracks", "Buildings"]):
    if tif_filename is None:
        tif_filename = "Unknown tif file"
        # print ("The generated polygons are for ", tif_filename)
    polygons = {
            "tif-slice-filename": tif_filename,
            "original-tif":"",
            "slice-georef": {
                "top":"",
                "left":"",
                "bottom":"",
                "right":""
                },
            "details" : {
                "Buildings": {},
                "Roads & Tracks": {},
                "Trees": {},
                "Crops": {},
                "Water": {}
                }
            }
    
    colors = {
        0: (150, 150, 150),  # Buildings
        1: (223, 194, 125),  # Roads & Tracks
        2: (27, 120, 55),    # Trees
        3: (166, 219, 160),  # Crops
        4: (116, 173, 209)   # Water
    }

    class_label= {
        0: "Buildings",
        1: "Roads & Tracks",
        2: "Trees",
        3: "Crops",
        4: "Water"
    }

    for cl, label in class_label.items():
        if label in reqd_class_label:
            pol = binary_mask_to_polygon(bin_mask_array[cl,:,:])
        # print ("There are {} polygons in class {}".format(len(pol), label))
        # TODO: Save images for visualization in separate function / module
        # im = Image.new("RGB", bin_mask_array.shape[1:])
        # draw = ImageDraw.Draw(im)
            for i in range(len(pol)):
                polygons["details"][label][str(i)] = pol[i]
            # draw.polygon(pol[i], outline=colors[cl], fill=colors[cl])
        # im.save("class_" + str(cl) +".jpg")
    # print (json.dumps(polygons, indent=4))
    
    with open(result_path + '/' + tif_filename + '_polys.json', 'w') as json_file:
        json.dump(polygons, json_file, indent=4)

