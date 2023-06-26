# Reference maps of soil phosphorus for the pan-Amazon region - CODE & DATA

## Introduction

This repository contains code and data employed to generate a set of reference maps of soil phosphorus (P) for the [Pan-Amazon](https://www.amazoniasocioambiental.org/pt-br/radar/raisg-lanca-colecao-2-0-do-mapbiomas-amazonia/) region. The primary intent of these maps is to provide reference data for parametrization and benchmark of Land Surface/Terrestrial Ecosystem models.

The maps created are the mean prediction of a set of random forest regression models fitted with available observed _in_ _situ_ data found in scientific literature. The model predictions are generated based on data from geographic datasets that have the same features utilized to fit the models.

## Results

The final maps from REF 1 (The original experiment) are archived [here](./RESULTS/).

## Input data

We employed data from several other datasets as input to the P maps generation. These data may have other licensing than the [MIT Licence](https://opensource.org/licenses/MIT). The references are listed in the [README](./inputDATA/README.md).

## Reproducing the P maps

If you want it is possible to build the P maps archived in the ./RESULTS folder and create similar figures found in REF 1. The created figures are stored in a folder named ./p_figs

## Software dependencies

- python3 - numpy, pandas, matplotlib, cartopy, scikit-learn, netCDF4, cfunits

- make, geos, proj, udunits2

### 1 - **Create the maps and figures:**

Please, note that this program does a high amount of computations when configured to the original/full experiment. To test it and also to make a preliminary analysis of the methods I tweaked the initial number of models generated. In [FILE 1](./rforest_pfracs.py) the global variable NMODELS at line 22 can be changed at your will. In the [Makefile](./Makefile) the files that do de work are organized into the logical sequence of execution (FILE1 to FILE9) and can executed at command.

First, install software dependencies. I suppose that you have a python3 (called python) that you can call from the command line. The same for make. You can change the python executable in the first line of the [Makefile](./Makefile).

You can do it in a GNU/linux operating system (tested). In windows you can set up a environment with the required software using conda (anaconda3, tested). Not tested in other OS.

Navigate to the main folder:

``$ make pmaps``

Done.

The files with the maps are created in the root folder in NETCDF4(HDF5) format (The results of the original experiment are [here](./RESULTS/)). CRS=EPSG4326 (WGS84)

The masks generated by the calculation of the dissimilarity index are stored in [this folder](./dissimilarity_index_masks/).

The software was built incrementally during the developement of the maps. Some scripts use globbing to find data generated by the scripts executed before. Thus, there is a chain of events that need to happen in a ordered way. If you change the code or want to re-run the process, use ``$ make clean``  to delete the old files before the new execution.

## Creating an environment with conda (anaconda3) in windows

Issue the following command on the anaconda3 PS/cmd prompt to create a new virtual environment called pmaps. It will be used to run the code:

``(base)C:\> conda create --channel conda-forge -n pmaps make m2-base geos proj udunits2 python numpy pandas matplotlib cartopy scikit-learn netCDF4 cfunits``

At this point close the anaconda prompt and set the environment variable UDUNITS2_XML_PATH to the path of the udunits2.xml file in your system. This file will be in the user folder, under an address like ~/anaconda3/Library/share/udunits/udunits2.xml.

Re-open the anaconda prompt and navigate to the home folder of this README and activate the newly created virtual environment:

``(base)C:\> conda activate pmaps``

Then you just use make to make the P maps:

``(pmaps)C:\> make pmaps``

The chosen colormaps are colourblind-friendly. I am thankful to Fabio Crameri for providing the [Scientific Colormaps](https://zenodo.org/record/5501399).

### References

1 - Darela-filho et al. 202x, Reference maps of soil phosphorus for the pan-Amazon region. To be submited to ESSD.

2 - Crameri F, Shephard GE, Heron PJ. 2020. The misuse of colour in science communication. Nature Communications 11(1): 5444.
