from glob import glob1
from os import makedirs
from pathlib import Path
from sys import argv

import arcpy
from arcpy.sa import RasterCalculator, SetNull, ExtractByMask
from arcpy.ia import NbrRectangle, FocalStatistics, Con, IsNull

from owslib.wcs import WebCoverageService

import numpy as np


__descr__ = """download files from soilgrids using WCS (https://maps.isric.org/) and uses arcpy to do some pre-processing"""

try:
    dld = argv[1]
except:
    dld = 1

# Check out the ArcGIS Spatial Analyst extension license
arcpy.CheckOutExtension("Spatial")
arcpy.CheckOutExtension("ImageAnalyst")

output = Path("./data").resolve()
makedirs(output, exist_ok=True)

# Set the analysis environments
arcpy.env.workspace = str(output)
arcpy.env.overwriteOutput = True

mask = r"..\shp\pan_Amazon_mask.shp"
dump_folder = Path("../predictive_rasters_5m").resolve()

BBOX = [-80, -21, -42, 10]

filepaths = {}

map_servers = {"nitrogen":     "https://maps.isric.org/mapserv?map=/map/nitrogen.map",
               "bdod":         "https://maps.isric.org/mapserv?map=/map/bdod.map",
               "clay":         "https://maps.isric.org/mapserv?map=/map/clay.map",
               "sand":         "https://maps.isric.org/mapserv?map=/map/sand.map",
               "silt":         "https://maps.isric.org/mapserv?map=/map/silt.map",
               "soc":          "https://maps.isric.org/mapserv?map=/map/soc.map",
               "phh2o" :       "https://maps.isric.org/mapserv?map=/map/phh2o.map"
               }

conv_factors = {          # converts to:
    "nitrogen": 1/100/10, # %
    "bdod": 1/100,        # kg dm-3
    "clay": 1/10/100,     # fraction 0-1
    "sand": 1/10/100,     # fraction 0-1
    "silt": 1/10/100,     # fraction 0-1
    "soc": 1/10/10,       # %
    "phh2o": 1/10,        # -log(H+)
    }

new_name = {
    "nitrogen": "nitrogen",
    "bdod": "bulk_density",
    "clay": "clay",
    "sand": "sand",
    "silt": "silt",
    "soc": "toc",
    "phh2o": "ph",
    }

def get_service(vname):
    return WebCoverageService(map_servers[vname], version='1.0.0'), vname

def get_wcs(vname):
    wcs, vn = get_service(vname)
    names = [k for k in wcs.contents.keys() if k.find("0-5cm_mean") != -1]
    names.append([k.strip() for k in wcs.contents.keys() if k.find("5-15cm_mean") != -1][0])
    names.append([k.strip() for k in wcs.contents.keys() if k.find("15-30cm_mean") != -1][0])
    return wcs, names, vn

def get_files(vname):
    return glob1(output, f"{vname}_*"), vname

def download(vname):
    fpath = []

    if dld:
        wcs, names, vn = get_wcs(vname)
    else:
        names, vn = get_files(vname)

    for name in names:
        if dld:
            res = wcs.getCoverage(
                resx = 0.083333333333333333,
                resy = 0.083333333333333333,
                identifier= name,
                crs = "urn:ogc:def:crs:EPSG::4326",
                bbox = BBOX,
                format = 'GEOTIFF_INT16'
                )
            fname = output/Path(f"{name}.tif")
        else:
            fname = output/Path(f"{name}")

        fpath.append(fname)
        if dld:
            with open(fname, 'wb') as file:
                file.write(res.read())
    filepaths[vn] = fpath

def process(vname):

    assert len(filepaths) > 0
    assert vname in filepaths.keys()

    extraction_area = "INSIDE"
    analysis_extent = mask

    data = []
    for k in filepaths[vname]:
        data.append(str(k))

    cv = conv_factors[vname]
    op = f"((a*%.7f)+(b*%.7f)+(c*%.7f))/3" % (cv, cv, cv)

    out_rc = RasterCalculator(data, ["a", "b", "c"],
                                       op ,"IntersectionOf", "MinOf")
    # Set zero to nodata
    final_tif = str(dump_folder/Path(f"{new_name[vname]}.tif").resolve())
    if vname in list(map_servers.keys()):
        out_rc = SetNull(out_rc, out_rc, "VALUE = 0")
        # Focal stats to fill no data
        neighborhood = NbrRectangle(4, 4, "CELL")
        focal_stats = FocalStatistics(out_rc, neighborhood, "MEAN", "")
        focal_stats = ExtractByMask(focal_stats, mask, extraction_area, analysis_extent)
        # focal_stats.save(f"{vname}_focal_stats.tif")

    out_rc = ExtractByMask(out_rc, mask, extraction_area, analysis_extent)

    out_rc = Con(IsNull(out_rc) ,focal_stats, out_rc)

    arcpy.management.CopyRaster(in_raster=out_rc, out_rasterdataset=final_tif,
                                nodata_value="-9999", pixel_type="32_BIT_FLOAT", format="TIFF")

    out_rc.save(f"{vname}.tif")


def get_wrb():
    wsp = arcpy.env.workspace
    output = Path("./WRB").resolve()
    makedirs(output, exist_ok=True)
    arcpy.env.workspace = str(output)
    wcs = WebCoverageService("https://maps.isric.org/mapserv?map=/map/wrb.map", version="1.0.0")
    names = wcs.contents.keys()
    format = 'GEOTIFF_BYTE'
    extraction_area = "INSIDE"
    analysis_extent = mask
    for name in names:
        res = wcs.getCoverage(
                resx = 0.083333333333333333,
                resy = 0.083333333333333333,
                identifier= name,
                crs = "urn:ogc:def:crs:EPSG::4326",
                bbox = BBOX,
                format = format
                )
        fid = Path(f"{name}.tif")
        fname = output/fid
        with open(fname, 'wb') as file:
            file.write(res.read())

        out_extr = ExtractByMask(f"{name}.tif", mask, extraction_area, analysis_extent)

        if name != "MostProbable":
            op = "a/100.0"
            out_extr = RasterCalculator([out_extr,], ["a",],
                                       op ,"IntersectionOf", "MinOf")
        final_tif = str(dump_folder/fid)

        arcpy.management.CopyRaster(in_raster=out_extr, out_rasterdataset=final_tif,
                                nodata_value="-9999", pixel_type="32_BIT_FLOAT", format="TIFF")
        out_extr.save(f"{name}.tif")
    arcpy.env.workspace = wsp


if __name__ == "__main__":
    get_wrb()
    for key in map_servers.keys():
        download(key)
        process(key)
