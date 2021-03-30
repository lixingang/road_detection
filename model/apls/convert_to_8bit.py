import gdal
from apls.src.apls_tools import convert_to_8Bit
import numpy as np
import os
if __name__=='__main__':
    imgPath = '/home/data/lixg/Spacenet/AOI_2_Vegas_Roads_Test_Public/RGB-PanSharpen'
    outPath = 'outPath'
    for img in os.listdir(imgPath):
        in_img = os.path.join(imgPath, img)
        out_img = os.path.join(outPath, img)
        convert_to_8Bit(in_img, out_img,
                                outputPixType='Byte',
                                outputFormat='GTiff',
                                rescale_type='rescale',
                                percentiles=[2,98])
    
    # img = gdal.Open(imgPath)
    # nda = img.ReadAsArray()
    # nda = np.transpose(nda,[1,2,0])
    # print(nda.shape)
    # print(nda[:,:,2])