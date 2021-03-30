import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from tqdm import tqdm
import json
from osgeo import gdal,osr,ogr

ROOT_DIR = os.path.abspath("./")
sys.path.append(ROOT_DIR)
ROOT_DIR = os.path.abspath("./utils")
sys.path.append(ROOT_DIR)

import apls.utils.skeleton as ske
from apls.utils.toGeojson import *
import re
def find_name(s):
    pattern = re.compile(r'img\d{1,4}')
    match = pattern.search(s)
    if match:
        return match.group()
def save_linestring_file(filename, wkt):
    file = open(filename, 'w')
    file.writelines(["%s\n" % item for item in wkt])
    file.close()
def save_geojson_file(filename):
    pass

def toJson(pngname,inputRaster,o_json_path):
    # print(pngname)
    file = find_name(pngname)
    city, wkt = ske.build_graph_single(pngname, debug=False, threshes=0.2, \
                                       add_small=True, fix_borders=False,reverse=False,inputTiff=inputRaster)
    linestring_filename='/home/lxg/data/road/spacenet/logs_Vegas/logs_ours/apls_outputs/outputs_txt/'+'spacenetroads_AOI_2_Vegas_'+file+'.txt'
    geojson_filename=os.path.join(o_json_path,'spacenetroads_AOI_2_Vegas_'+file+'.geojson')
    save_linestring_file(linestring_filename,wkt)
    js = toGeojson(linestring_filename)
    with open(geojson_filename ,'w') as f:
        json.dump(js.get_geojson(),f)



tif_root = os.path.join("/home/lxg/data/road/spacenet/vegas/images_rgb_1300/")
def batch_toJson_prop(root='/home/lxg/data/road/spacenet/logs_Vegas/logs_ours/main_outputs/'):
    png_root = os.path.join(root,'preds_threshold')
    i = 0
    for pngname in tqdm(os.listdir(png_root)):
        if pngname.split('.')[-1]!='png':
            continue
        imgNo = find_name(pngname)
        imgRaster = os.path.join(tif_root, 'RGB-PanSharpen_AOI_2_Vegas_'+imgNo+'.tif')
        toJson(os.path.join(png_root, pngname), imgRaster,'/home/lxg/data/road/spacenet/logs_Vegas/logs_ours/apls_outputs/prop_geojson/')
        i+=1
    print('[I] batch_toJson_prop: Generate Geojson Successfully',i)

def batch_toJson_label(dataroot='/home/lxg/data/road/spacenet/logs_Vegas/logs_ours/main_outputs'):
    png_root = os.path.join(dataroot,'labels')
    i = 0
    for pngname in tqdm(os.listdir(png_root)):
        if pngname.split('.')[-1]!='png':
            continue

        imgNo = find_name(pngname)
        # print(imgNo)
        imgRaster = os.path.join(tif_root, 'RGB-PanSharpen_AOI_2_Vegas_'+imgNo+'.tif')
        toJson(os.path.join(png_root, pngname), imgRaster,'/home/lxg/data/road/spacenet/logs_Vegas/logs_ours/apls_outputs/labels_geojson/')
        i+=1
    print('[I] batch_toJson_label:Generate Geojson Successfully',i)
