

#### Make a copy

# Import system modules
import arcpy

# Set workspace
arcpy.env.workspace = "C:/data"

# Set local variables
in_data =  "majorrds.shp"
out_data = "C:/output/majorrdsCopy.shp"

# Run Copy
arcpy.management.Copy(in_data, out_data)



#### extract values from rasters


# Name: ExtractMultiValuesToPoints_Ex_02.py
# Description: Extracts the cells of multiple rasters as attributes in
#    an output point feature class.  This example takes a multiband IMG
#    and two GRID files as input.
# Requirements: Spatial Analyst Extension

# Import system modules
import arcpy
from arcpy import env
from arcpy.sa import *

# Set environment settings
env.workspace = "C:/sapyexamples/data"

# Set local variables
inPointFeatures = "poi.shp"
inRasterList = [["doqq.img", "doqqval"], ["redstd", "focalstd"],
                ["redmin", "focalmin"]]

# Execute ExtractValuesToPoints
ExtractMultiValuesToPoints(inPointFeatures, inRasterList, "BILINEAR")


######  remove nulls

import arcpy

# Assuming 'your_shapefile' is your shapefile and 'fields' is a list of your fields
your_shapefile = 'path_to_your_shapefile'
fields = ['field1', 'field2', 'field3', ...]

# Create a feature layer from the shapefile
arcpy.MakeFeatureLayer_management(your_shapefile, "lyr")

for field in fields:
    # SQL to select non-nulls
    fieldDelim = arcpy.AddFieldDelimiters("lyr", field)
    sql = "{} IS NOT NULL".format(fieldDelim)

    # Select non-nulls
    arcpy.SelectLayerByAttribute_management("lyr", "NEW_SELECTION", sql)

# Export the selected features to a new shapefile
arcpy.CopyFeatures_management("lyr", 'path_to_new_shapefile')

# Delete the feature layer to clean up
arcpy.Delete_management("lyr")



### export to csv

import arcpy
arcpy.env.workspace = "C:/data"
arcpy.conversion.ExportTable("vegtable.dbf", "C:/output/output.gdb/vegtable")

