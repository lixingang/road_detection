'''
LINESTRING (441.0 1.0, 350.0 4.0, 180.0 3.0, 165.0 16.0, 167.0 32.0, 186.0 51.0, 186.0 161.0)
LINESTRING (165.0 16.0, 144.0 0.0, 139.0 0.0, 120.0 19.0, 133.0 31.0, 138.0 45.0, 138.0 141.0, 156.0 158.0)
->
{
"type": "FeatureCollection",
"crs": { "type": "name", "properties": { "name": "urn:ogc:def:crs:OGC:1.3:CRS84" } },
"features": [
{ "type": "Feature", "properties": { "id": 0 }, "geometry": { "type": "LineString", "coordinates": [ [ -115.303514445, 36.1969076699 ], [ -115.30134526499999, 36.1968795899 ] ] } },
{ "type": "Feature", "properties": { "id": 1 }, "geometry": { "type": "LineString", "coordinates": [ [ -115.30134877499999, 36.196844489900002 ], [ -115.30134526499999, 36.195931889900002 ] ] } }
]
}
'''
import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import json
import re
import geojson
import gdal
import shapely.wkt
from osgeo import gdal
import json
class toGeojson:
    def __init__(self,myfile):
        with open(myfile, 'r') as file:
            self.listenlines = file.readlines()
            self.dt = {}

        self._fill_dict()
        # print(truthGeoJson.loads(self.dt))
    def get_geojson(self, save=False):
        if save:
            with open(save, 'w') as result_file:
                json.dump(self.dt, result_file)
        else:
            return self.dt
    


    @staticmethod
    def _get_geometry(listenlines,DEBUG=False):
        # return {"coordinates":[[],[],[]],"type":"LineString"}
        g1 = shapely.wkt.loads(listenlines)
        g2 = geojson.Feature(geometry=g1, properties={})
        if DEBUG:
            print(g2.geometry)
        return g2

    def _fill_dict(self):
        self.dt["type"] = "FeatureCollection"
        #self.dt["crs"] = {}; self.dt["crs"]["type"] = "name"; #self.dt["crs"]["properties"] = {}
        #self.dt["crs"]["properties"]["name"] = "urn:ogc:def:crs:OGC:1.3:CRS84"
        self.dt["features"] = []

        for id,line in enumerate(self.listenlines):
            dt_feature = {}
            dt_feature["type"] = "Feature"
            dt_feature["properties"] = {}
            dt_feature["properties"]["id"] = str(id)
            dt_feature["geometry"] = self._get_geometry(line).geometry
            self.dt["features"].append(dt_feature)


if __name__=='__main__':
    worker = toGeojson('out.txt')
    print(worker.get_geojson('result.txt'))
    #rker._readTif('../data/RGB-PanSharpen_AOI_2_Vegas_img1326.tif')

