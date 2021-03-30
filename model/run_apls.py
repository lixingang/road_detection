import os,sys
ROOT_DIR = os.path.abspath("./")
sys.path.append(ROOT_DIR)
ROOT_DIR = os.path.abspath("./apls")
sys.path.append(ROOT_DIR)
ROOT_DIR = os.path.abspath("./apls/utils")
sys.path.append(ROOT_DIR)

from apls.generate_geojson import *
import apls.apls.apls as apls

batch_toJson_label() # 'labels_pixel_png'
batch_toJson_prop() # 'images'
apls.main()
print('Done')
