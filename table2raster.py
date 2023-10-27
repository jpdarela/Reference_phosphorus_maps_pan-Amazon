# -*- coding: utf-8 -*-
"""THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
   EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
   OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
   IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR
   ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
   TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
   SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
from math import sqrt
import concurrent.futures
from os import makedirs
import glob
from pathlib import Path

from netCDF4 import Dataset
import cfunits
import numpy as np
import pandas as pd

from utilities import PFRACS, find_coord

FILE = 3

"""Create netCDF4 files with the predicted P forms for each model for each P form
    USAGE: python table2raster.py
    SIDE EFFECTS: Create netCDF4 files named predicted_<pform>.nc4 """

comment = "Values for concentration and for area density estimates of P are for the 0-30 cm soil layer, i.e. the topsoil."

geo_description = np.load("./inputDATA/extent.npz")
area = geo_description["area"] * (-1)
mask = geo_description["mask"]
lat = geo_description["lat"]
lon = geo_description["lon"]
cell_size = geo_description["cell_size"]

ydim = geo_description["lat"].size
xdim = geo_description["lon"].size

outnc = Path("./results/")
makedirs(outnc, exist_ok=True)

# Soil dry bulk density from SoilGrids
bd = Dataset("./inputDATA/bulk_density.nc").variables['bulk_density'][:]


def rasterize(arr, clean=True, offset=0):
    out = np.zeros(shape=(ydim, xdim), dtype=np.float32) - 9999.0
    laty = arr[:, 1 - offset]
    lonx = arr[:, 2 - offset]
    val =  arr[:, 3 - offset]
    if clean:
        cut_point = np.percentile(val, 99.85)
        idx = np.where(val > cut_point)
        val[idx[0]] = -9999.0
    for y, x, v in zip(laty, lonx, val):
        ny, nx = find_coord(y, x, cell_size, lat, lon)
        out[ny, nx] = v
    return out


def save_nc(nc_filename, arr, varname, ndim=None, axis_data=[0,], units='mg.kg-1', comm=comment):

    rootgrp = Dataset(nc_filename, mode='w', format='NETCDF4')

    la = arr.shape[0]
    lo = arr.shape[1]

    if ndim:
        la = arr.shape[1]
        lo = arr.shape[2]
        rootgrp.createDimension("models", ndim)

        #dimensions
    rootgrp.createDimension("latitude", la)
    rootgrp.createDimension("longitude", lo)
        #variables

    latitude = rootgrp.createVariable(varname="latitude",
                                      datatype=np.float32,
                                      dimensions=("latitude",))

    longitude = rootgrp.createVariable(varname="longitude",
                                       datatype=np.float32,
                                       dimensions=("longitude",))
    if ndim is not None:
        model = rootgrp.createVariable(varname="models",
                                   datatype=np.int32,
                                   dimensions=("models",))


        var_ = rootgrp.createVariable(varname = varname,
                                  datatype=np.float32,
                                  dimensions=("models", "latitude", "longitude",),
                                  fill_value=-9999.0,
                                  compression="zlib",
                                  complevel=9,
                                  shuffle=True,
                                  least_significant_digit=1,
                                  fletcher32=True)
    else:
        var_ = rootgrp.createVariable(varname = varname,
                                  datatype=np.float32,
                                  dimensions=("latitude","longitude",),
                                  fill_value=-9999.0)

        #attributes
        ## rootgrp
    rootgrp.description = 'Phosphorus forms in topsoil (0-30 cm) predicted by Random Forest regressions'
    rootgrp.source = "Reference Maps of Soil Phosphorus for the Pan-Amazon Region"
    rootgrp.author = "Joao Paulo Darela Filho<darelafilho@gmail.com>"
    rootgrp.info = comm

    ## lat
    latitude.units = u"degrees_north"
    latitude.long_name=u"latitude"
    latitude.standart_name =u"latitude"
    latitude.axis = u'Y'

    ## lon
    longitude.units = "degrees_east"
    longitude.long_name = "longitude"
    longitude.standart_name = "longitude"
    longitude.axis = 'X'

    ## models
    if ndim is not None:
        model.units = u"RANDOM_STATE"
        model.long_name = u"Selected models"
        model.axis = "T"

    ## var
    # var_.long_name = long_name[varname]
    var_.units = units
    var_.standard_name= varname
    var_.missing_value=-9999.0

    ## WRITING DATA
    longitude[:] = lon
    latitude[:] =  lat

    if ndim is not None:
        model[:] = np.array(axis_data, dtype=np.int32)
        var_[:, :, :] = arr
    else:
        var_[:,:] = arr
    rootgrp.close()


def ppools_gm2(vn, bd):
    """Build Netcdfs with values in g.m⁻²(0-300mm). Use the Dry Bulk density from SoilGrids"""
    assert vn in PFRACS or vn == "mineral_p"
    vname = vn
    if vname != "mineral_p":
        form_p = Dataset(outnc/Path(f"{vname}_AVG.nc")).variables[vname][:]
    else:
        form_p = Dataset(outnc/Path(f"mineral_p.nc")).variables["mineral_p"][:]

    tp = cfunits.Units.conform(form_p, cfunits.Units('mg kg-1'), cfunits.Units('g g-1'))
    den = cfunits.Units.conform(bd, cfunits.Units('kg dm-3'), cfunits.Units('g m-3'))

    p_form = 0.3 * tp * den
    save_nc(outnc/Path(f"{vname}_area_density.nc"), p_form, vname, ndim=None, units='g.m-2') # 0-30 cm


def process(label_name):
    assert label_name in PFRACS
    print(label_name)

    output = Path(f"predicted_P_{label_name}").resolve()
    files = glob.glob1(output, "*.feather")
    mdim = len(files)

    out = np.zeros(shape=(mdim, ydim, xdim), dtype=np.float32) - 9999.0
    model_r_states = []

    for i, file in enumerate(files):
        model_r_states.append(file.split('.')[0].split('_')[-1])
        arr = pd.read_feather(output/Path(file)).__array__()
        out[i, :, :]= rasterize(arr)

    # Save the predictions of all models
    data = np.ma.masked_array(out, out == -9999)
    save_nc(f"{outnc/Path(f'{label_name}.nc')}", data, label_name, ndim=mdim, axis_data=model_r_states)

    # SAve the mean prediction and store for use in the primary mineral P estimation
    pAVG = data.mean(axis=0,)
    save_nc(f"{outnc/Path(f'{label_name}_AVG.nc')}", pAVG, label_name)
    # SD & SE
    pSD = data.std(axis=0,)
    save_nc(f"{outnc/Path(f'{label_name}_SD.nc')}", pSD, label_name)
    pSE = pSD/sqrt(mdim)
    save_nc(f"{outnc/Path(f'{label_name}_SE.nc')}", pSE, label_name)

    #Estimate area density pools
    ppools_gm2(label_name, bd)


    return (label_name, pAVG)


if __name__ == "__main__":

    with concurrent.futures.ProcessPoolExecutor() as executor:
        result = executor.map(process, PFRACS)

    # Calculate the missing primary mineral P
    pforms = dict(list(result))
    mineral_p = pforms["total_p"] - (pforms["avail_p"] + pforms["org_p"] + pforms["inorg_p"] + pforms["occ_p"])
    pforms["mineral_p"] = mineral_p
    save_nc(f"{outnc/Path('mineral_p.nc')}", mineral_p, "mineral_p")
    ppools_gm2("mineral_p", bd)

    # calculate percenteges of the total
    for var in PFRACS[:4] + ["mineral_p",]:
        percentage = (pforms[var] / pforms["total_p"]) * 100.0
        save_nc(f"{outnc/Path(f'{var}_percentage_of_total_p.nc')}", percentage, var, units="%")

    save_nc("./inputDATA/area.nc", area, "area", units="m2")

