# -*- coding: utf-8 -*-
"""THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
   EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
   OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
   IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR
   ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
   TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
   SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import os
from glob import glob1
from math import sqrt
import numpy as np
from netCDF4 import Dataset
import pandas as pd
from table2raster import save_nc

FILE = 4

"""Calculate MEANS, SD and proportions of P fractions"""


def open_dataset(vname, fun='mean'):
    """Open the dataset and calculate mean or SD"""

    fname = "predicted_" + vname + ".nc4"

    fh = Dataset(fname)

    var = np.fliplr(fh.variables[vname][:])

    fh.close()
    if fun == 'mean':
        return var.mean(axis=0,)
    elif fun == 'std':
        return var.std(axis=0,)
    elif fun == 'se':
        scores = pd.read_csv('model_selection_scores.csv', index_col='pform')
        nmodels = scores.loc[[vname]]['nmodels']
        return var.std(axis=0,)/sqrt(nmodels)


def calculate_mineral():
    """ Estimate mineral + occluded pools"""

    org =  open_dataset("org_p")
    inorg =open_dataset("inorg_p")
    avail =open_dataset("avail_p")
    total =open_dataset("total_p")

    # TODO
    ## Quantificar a variabilidade/incerteza

    return total - (org + inorg + avail)

if __name__ == "__main__":

    files = glob1(os.getcwd(), "*.nc4")

    total_p = open_dataset('total_p')

    for fh in files:

        n = 10
        varn = fh[n:].split('.')[0]
        print(varn)

        # CALCULATE THE CONFIDENCE
        # All ensemble show high agreement among models - mean > 2 SD

        MEAN = open_dataset(varn)
        SD = open_dataset(varn, fun='std')
        SE = open_dataset(varn, fun='se')

        # High confidence
        # result = MEAN > (4.0 * SD)

        fname = varn + "_AVG.nc4"
        save_nc(fname, MEAN, varn, ndim=None)

        fname = varn + "_STD.nc4"
        save_nc(fname, SD, varn + "_STD", ndim=None)

        fname = varn + "_SE.nc4"
        save_nc(fname, SE, varn + "_SE", ndim=None)


        # CALCULATE THE PERCENNTAGES OF TOTAL P FOR OTHER FRACTIONS
        if varn == 'total_p':
            pass
        else:
            fname = varn + "_percent_of_tot.nc4"
            dt = open_dataset(varn)
            result = (dt / total_p) * 100.0
            save_nc(fname, result, varn + "percent_of_total_p", ndim=None)

    mineral = calculate_mineral()
    fname = 'Pmin1_Pocc.nc4'
    save_nc(fname, mineral, "mineral_p", ndim=None)
    prop = (mineral / total_p) * 100.0
    fname = "predicted_min-occ_over_total_p.nc4"
    save_nc(fname, prop, 'min1-occ_' + "percent_of_total_p", ndim=None)
