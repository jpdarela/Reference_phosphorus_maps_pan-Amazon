# Script to download data from SoilGrids using WCS

This script uses owslib to download the variables:

- WRB reference soil groups
- Sand, silt and clay fractions
- Total organic carbon and total nitrogen
- Soil bulk density

From [here](https://maps.isric.org/).

You can easily inspect the variables [here](https://soilgrids.org).

After the download some processing was made using arcpy. Rasters were masked using the [RAISG, 2022 - Amazon Limits](https://www.raisg.org/en/maps/) shapefile.

Annual mean temperature & precipitation, and elevation were downloaded from [WorldClim](https://worldclim.org/).

## References

### SoilGrids

Poggio, L., de Sousa, L. M., Batjes, N. H., Heuvelink, G. B. M., Kempen, B., Ribeiro, E., and Rossiter, D.: Soilgrids 2.0: Producing Soil Information for the Globe with Quantified Spatial Uncertainty, SOIL, 7, 217-240, [https://doi.org/10.5194/soil-7-217-2021](https://doi.org/10.5194/soil-7-217-2021), 2021.

### WorldClim

Fick, S. E. and Hijmans, R. J.: Worldclim 2: New 1-Km Spatial Resolution Climate Surfaces for Global Land Areas, Int J Climatol, 37, 4302-4315, [https://doi.org/10.1002/joc.5086](https://doi.org/10.1002/joc.5086), 2017.

### RAISG

RAISG - Amazon Network of Georeferenced Socio-Environmental Information: [https://www.raisg.org/en/about/](https://www.raisg.org/en/about/), last access:  October 2023.
