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
import numpy as np
import pandas as pd

FILE = 3

"""Create netCDF4 files with the predicted P forms for each model for each P form
    USAGE: python table2raster.py
    SIDE EFFECTS: Create netCDF4 files named predicted_<pform>.nc4 """

feat_list = ["lat","lon","Sand","Silt","Clay","Slope","Elevation","MAT","MAP","pH","TOC",
    "TN","SRG_Acrisols","SRG_Alisols","SRG_Andosols","SRG_Arenosols","SRG_Cambisols",
    "SRG_Ferralsols","SRG_Fluvisols","SRG_Gleysols","SRG_Lixisols","SRG_Luvisols","SRG_Nitisols",
    "SRG_Plinthosols","SRG_Podzol","SRG_Regosols","SRG_Umbrisols"]

diff=''

def col2arr(filename, varn, names):

    datan = pd.read_csv(filename)
    indexn = np.arange(datan.shape[0],dtype=np.int32)
    
    outn = np.zeros(shape=(360,720),dtype=np.float32) - 9999.0

    for i in indexn:
        nx = datan.iloc[i]['nx']
        ny = datan.iloc[i]['ny']
        data = datan.iloc[i][names[varn]]
        outn[np.int32(ny), np.int32(nx)] = data
            
    return np.ma.masked_array(outn, outn == -9999.0, fill_value=-9999.0), names[varn]


def save_nc(fname, arr, varname, ndim=None, axis_data=[0,], units='mg.kg-1'):
    
    from netCDF4 import Dataset
    var = ['total_p', 'org_p', 'inorg_p', 'avail_p', 
           'total_p_SE', 'org_p_SE', 'inorg_p_SE', 'avail_p_SE', 
           'total_p_STD', 'org_p_STD', 'inorg_p_STD', 'avail_p_STD',
           'org_ppercent_of_total_p', 'inorg_ppercent_of_total_p', 'avail_ppercent_of_total_p',
           'min1-occ_percent_of_total_p', 'mineral_p']
    lab = ['Total P', 'Organic P', 'Secondary Mineral P', 'Available P', 
           'Standard Error Total P', 'Standard Error Organic P', 'Standard Error Secondary Mineral P', 'Standard Error Available P',
           'Standard Deviation Total P', 'Standard Deviation Organic P', 'Standard Deviation Secondary Mineral P', 'Standard Deviation Available P',
           'Fraction of Total P - Organic P', 'Fraction of Total P - Secondary Mineral P', 'Fraction of Total P - Available P',
           'Fraction of Total P - Estimated Primary Mineral P + Occluded P', 'Estimated Primary Mineral P + Occluded P']
    
    long_name = dict(list(zip(var, lab)))
    
    nc_filename = fname

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
                                  dimensions=("models", "latitude","longitude",),
                                  fill_value=-9999.0)
    else:
        var_ = rootgrp.createVariable(varname = varname,
                                  datatype=np.float32,
                                  dimensions=("latitude","longitude",),
                                  fill_value=-9999.0)

        #attributes
        ## rootgrp
    rootgrp.description =  'Phosphorus forms in topsoil (0-30 cm) predicted by Random Forest regressions'
    rootgrp.source = "Reference Maps of Soil Phosphorus for the Pan-Amazon Region"
    rootgrp.author = "Joao Paulo Darela Filho<darelafilho@gmail.com>"
    
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
        model.long_name = u"Selected models (Accuracy >= 70%)"
        model.axis = "T"

    ## var
    var_.long_name = long_name[varname]
    var_.units = units
    var_.standard_name= varname
    var_.missing_value=-9999.0
    
    ## WRITING DATA
    longitude[:] = np.arange(-179.75, 180, 0.5)
    latitude[:] =  np.arange(-89.75, 90, 0.5)

    if ndim is not None:
        model[:] = np.array(axis_data, dtype=np.int32)
        var_[:, :, :] = np.fliplr(arr)
    else:
        var_[:,:] = np.flipud(arr)
    rootgrp.close()

if __name__ == "__main__":
    pfracs = ["inorg_p", "org_p", "avail_p", "total_p"]
    
    for label_name in pfracs:
        print(label_name)
        names = feat_list + [label_name,]
        folder = "./predicted_P_%s/" % label_name
        files = glob.glob1(folder, "*.csv")
        nmodels = int(len(files))
        output_arr = np.zeros(shape=(nmodels,360,720),dtype=np.float32) - 9999.0
        model_r_states = []
        for i, fh in enumerate(files):
            filename_store = fh
            model_r_states.append(fh.split('.')[0].split('_')[-1])
            arr = col2arr(folder + fh, -1, names)
            output_arr[i,:,:] = arr[0].__array__()

        output_arr = np.ma.masked_array(output_arr, output_arr == -9999.0)

        frac = filename_store.split('.')[0].split('_')[1]
        de_que = filename_store.split('.')[0].split('_')[2]

        fname = filename_store.split('.')[0].split('_')[0] + diff + "_" + frac + "_" + de_que + ".nc4"
        save_nc(fname, output_arr, label_name, ndim=len(model_r_states), axis_data=model_r_states)
