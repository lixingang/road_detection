from osgeo import gdal,osr
import numpy as np
def imagexy2geo(dataset, row, col):
    trans = dataset.GetGeoTransform()
    px = trans[0] + col*trans[1]+row*trans[2]
    py = trans[3] + col*trans[4] + row*trans[5]
    return [px,py]
def array2raster(outpath, array, geoTransform, proj):
    cols = array.shape[1]
    rows = array.shape[0]
    driver = gdal.GetDriverByName('Gtiff')
    outRaster = driver.Create(newRasterfn,cols,rows,1,gdal.GDT_Byte)
    outRaster.SetGeoTransform(geoTransform)
    outband = outRaster.GetRasterBand(1)
    outband.WriteArray(array)
    outRaster.SetProjection(proj)
    outRaster.FlushCache()
if __name__ == '__main__':
    gdal.AllRegister()
    dataset = gdal.Open('data/RGB-PanSharpen_AOI_2_Vegas_img1326.tif')
    pngimage =
    coords = imagexy2geo(dataset, 0,0)
    print(coords)