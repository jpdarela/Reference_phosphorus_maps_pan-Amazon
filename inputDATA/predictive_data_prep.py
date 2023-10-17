import glob

from pathlib import Path

import arcpy
from arcpy import env
from arcpy.sa import *


#### Make a copy of the shapefile that will be employed to extract pixel values

# Set workspace
raster_folder = Path("./predictive_rasters_5m")

env.workspace = str(raster_folder)
env.overwriteOutput = True

# Set global variables

template_shp = Path("./shp/template_predictive_dataset.shp")

predictive_shp = r"predictive.shp"

def prepare_shp():
    # Run Copy
    arcpy.management.Copy(str(template_shp), predictive_shp)

    # Extract multi values to points
    in_rasters = [
        ["sand.tif", "Sand"],
        ["silt.tif", "Silt"],
        ["clay.tif", "Clay"],
        ["slope.tif", "Slope"],
        ["elevation.tif", "Elevation"],
        ["mat.tif", "MAT"],
        ["map.tif", "MAP"],
        ["ph.tif", "pH"],
        ["toc.tif", "TOC"],
        ["nitrogen.tif", "TN"],
        ["Acrisols.tif", "Acrisols"],
        ["Alisols.tif", "Alisols"],
        ["Andosols.tif", "Andosols"],
        ["Arenosols.tif", "Arenosols"],
        ["Cambisols.tif", "Cambisols"],
        ["Ferralsols.tif", "Ferralsols"],
        ["Fluvisols.tif", "Fluvisols"],
        ["Gleysols.tif", "Gleysols"],
        ["Lixisols.tif", "Lixisols"],
        ["Luvisols.tif", "Luvisols"],
        ["Nitisols.tif", "Nitisols"],
        ["Plinthosols.tif", "Plinthosols"],
        ["Podzols.tif", "Podzols"],
        ["Regosols.tif", "Regosols"],
        ["Umbrisols.tif", "Umbrisols"]]

    # Assert files exists
    for a in in_rasters:
        raster = raster_folder/Path(a[0])
        assert raster.exists(), f"{str(raster)} not found"

    # # Execute ExtractValuesToPoints
    ExtractMultiValuesToPoints(predictive_shp, in_rasters, None)

prepare_shp()

# # Create a feature layer from the shapefile
arcpy.MakeFeatureLayer_management(predictive_shp, "lyr")

WHERE = "Sand = -9999 Or Silt = -9999 Or Clay = -9999 Or Slope = -9999 Or Elevation = -9999 Or MAT = -9999 Or MAP = -9999 Or pH = -9999 Or TOC = -9999 Or TN = -9999"

# "Sand = 0 And Silt = 0 And Clay = 0 Or Slope = -9999 Or Elevation = -9999 Or MAT = -9999 Or MAP = -9999 Or pH = 0 Or TOC = 0 Or TOC = -9999 Or TN = 0 Or TN = -9999"

arcpy.SelectLayerByAttribute_management("lyr", "NEW_SELECTION", WHERE)
arcpy.SelectLayerByAttribute_management("lyr", "SWITCH_SELECTION")

# Export the selected features to a new shapefile
arcpy.CopyFeatures_management("lyr", 'predictive_final.shp')

# Delete the feature layer to clean up
arcpy.Delete_management("lyr")

# Export to csv
arcpy.conversion.ExportTable('predictive_final.shp', "../inputDATA/predictive.csv")


