# -*- coding: utf-8 -*-
"""THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
   EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
   OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
   IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR
   ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
   TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
   SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import glob
from pathlib import Path
from numba import jit

from netCDF4 import Dataset
import cfunits
import numpy as np
import pandas as pd

from utilities import PFRACS, find_coord

FILE = 3

"""Create netCDF4 files with the predicted P forms for each model for each P form
    USAGE: python table2raster.py
    SIDE EFFECTS: Create netCDF4 files named predicted_<pform>.nc4 """

geo_description = np.load("./inputDATA/extent.npz")
lat = geo_description["lat"]
lon = geo_description["lon"]
cell_size = geo_description["cell_size"]

ydim = geo_description["lat"].size
xdim = geo_description["lon"].size


def rasterize(arr):
    out = np.zeros(shape=(ydim, xdim), dtype=np.float32) - 9999.0
    laty = arr[:,1]
    lonx = arr[:,2]
    val = arr[:,3]
    cut_point = np.percentile(val, 99.85)
    idx = np.where(val > cut_point)
    val[idx[0]] = -9999.0
    for y, x, v in zip(laty, lonx, val):
        ny, nx = find_coord(y, x, cell_size, lat, lon)
        out[ny, nx] = v
    return out

# def save_nc(fname, arr, varname, ndim=None, axis_data=[0,], units='mg.kg-1'):

#     from netCDF4 import Dataset
#     var = ['total_p', 'org_p', 'inorg_p', 'avail_p',
#            'total_p_SE', 'org_p_SE', 'inorg_p_SE', 'avail_p_SE',
#            'total_p_STD', 'org_p_STD', 'inorg_p_STD', 'avail_p_STD',
#            'org_ppercent_of_total_p', 'inorg_ppercent_of_total_p', 'avail_ppercent_of_total_p',
#            'min1-occ_percent_of_total_p', 'mineral_p']
#     lab = ['Total P', 'Organic P', 'Secondary Mineral P', 'Available P',
#            'Standard Error Total P', 'Standard Error Organic P', 'Standard Error Secondary Mineral P', 'Standard Error Available P',
#            'Standard Deviation Total P', 'Standard Deviation Organic P', 'Standard Deviation Secondary Mineral P', 'Standard Deviation Available P',
#            'Fraction of Total P - Organic P', 'Fraction of Total P - Secondary Mineral P', 'Fraction of Total P - Available P',
#            'Fraction of Total P - Estimated Primary Mineral P + Occluded P', 'Estimated Primary Mineral P + Occluded P']

#     long_name = dict(list(zip(var, lab)))

#     nc_filename = fname

#     rootgrp = Dataset(nc_filename, mode='w', format='NETCDF4')

#     la = arr.shape[0]
#     lo = arr.shape[1]

#     if ndim:
#         la = arr.shape[1]
#         lo = arr.shape[2]
#         rootgrp.createDimension("models", ndim)

#         #dimensions
#     rootgrp.createDimension("latitude", la)
#     rootgrp.createDimension("longitude", lo)
#         #variables

#     latitude = rootgrp.createVariable(varname="latitude",
#                                       datatype=np.float32,
#                                       dimensions=("latitude",))

#     longitude = rootgrp.createVariable(varname="longitude",
#                                        datatype=np.float32,
#                                        dimensions=("longitude",))
#     if ndim is not None:
#         model = rootgrp.createVariable(varname="models",
#                                    datatype=np.int32,
#                                    dimensions=("models",))


#         var_ = rootgrp.createVariable(varname = varname,
#                                   datatype=np.float32,
#                                   dimensions=("models", "latitude","longitude",),
#                                   fill_value=-9999.0)
#     else:
#         var_ = rootgrp.createVariable(varname = varname,
#                                   datatype=np.float32,
#                                   dimensions=("latitude","longitude",),
#                                   fill_value=-9999.0)

#         #attributes
#         ## rootgrp
#     rootgrp.description =  'Phosphorus forms in topsoil (0-30 cm) predicted by Random Forest regressions'
#     rootgrp.source = "Reference Maps of Soil Phosphorus for the Pan-Amazon Region"
#     rootgrp.author = "Joao Paulo Darela Filho<darelafilho@gmail.com>"

#     ## lat
#     latitude.units = u"degrees_north"
#     latitude.long_name=u"latitude"
#     latitude.standart_name =u"latitude"
#     latitude.axis = u'Y'

#     ## lon
#     longitude.units = "degrees_east"
#     longitude.long_name = "longitude"
#     longitude.standart_name = "longitude"
#     longitude.axis = 'X'

#     ## models
#     if ndim is not None:
#         model.units = u"RANDOM_STATE"
#         model.long_name = u"Selected models (Accuracy >= 70%)"
#         model.axis = "T"

#     ## var
#     var_.long_name = long_name[varname]
#     var_.units = units
#     var_.standard_name= varname
#     var_.missing_value=-9999.0

#     ## WRITING DATA
#     longitude[:] = np.arange(-179.75, 180, 0.5)
#     latitude[:] =  np.arange(-89.75, 90, 0.5)

#     if ndim is not None:
#         model[:] = np.array(axis_data, dtype=np.int32)
#         var_[:, :, :] = np.fliplr(arr)
#     else:
#         var_[:,:] = np.flipud(arr)
#     rootgrp.close()

def ppools_gm2(vname):
    """Build Netcdfs with values in g.m⁻²(0-300mm). Use the Dry Bulk density from SoilGrids"""
    var = PFRACS
    if vname in var:
        dt = Dataset(f"./predicted_{vname}.nc4").variables[vname][:,:,:]
        form_p = np.flipud(dt.data.mean(axis=0,))
        mask = form_p == -9999.0
    else:
        dt = Dataset("./Pmin1_Pocc.nc4").variables[vname][:,:]
        form_p = np.flipud(dt.data)
        mask = form_p == -9999.0

    bd = Dataset("./inputDATA/soil_bulk_density.nc").variables['b_ds_final'][:]

    tp = cfunits.Units.conform(form_p, cfunits.Units('mg kg-1'), cfunits.Units('g g-1'))
    den = cfunits.Units.conform(bd, cfunits.Units('kg dm-3'), cfunits.Units('g m-3'))

    p_form = np.ma.masked_array((0.3 * tp) * den, mask=mask == True)
    save_nc(f"{vname}_density.nc4", p_form, vname, ndim=None, units='g.m-2')

if __name__ == "__main__":

    for label_name in PFRACS:
        print(label_name)
        output = Path(f"predicted_P_{label_name}").resolve()
        files = glob.glob1(output, "*.feather")
        mdim = len(files)
        out = np.zeros(shape=(mdim, ydim, xdim), dtype=np.float32)
        for i, file in enumerate(files):
            arr = pd.read_feather(output/Path(file)).__array__()
            out[i, :, :]= rasterize(arr)
        # exclude gridcells with high SE here
        data = np.ma.masked_array(out, out == -9999)
        #:#save netCDF
        bulk_density = ""

        # stats
        pAVG = ""
        pSE = ""
        pSD = ""

        # area desnsity





        # names = feat_list + [label_name,]
        # folder = "./predicted_P_%s/" % label_name
        # files = glob.glob1(folder, "*.csv")
        # nmodels = int(len(files))
        # output_arr = np.zeros(shape=(nmodels,360,720),dtype=np.float32) - 9999.0
        # model_r_states = []
        # for i, fh in enumerate(files):
        #     filename_store = fh
        #     model_r_states.append(fh.split('.')[0].split('_')[-1])
        #     arr = col2arr(folder + fh, -1, names)
        #     output_arr[i,:,:] = arr[0].__array__()

        # output_arr = np.ma.masked_array(output_arr, output_arr == -9999.0)

        # frac = filename_store.split('.')[0].split('_')[1]
        # de_que = filename_store.split('.')[0].split('_')[2]

        # fname = filename_store.split('.')[0].split('_')[0] + diff + "_" + frac + "_" + de_que + ".nc4"
        # save_nc(fname, output_arr, label_name, ndim=len(model_r_states), axis_data=model_r_states)
