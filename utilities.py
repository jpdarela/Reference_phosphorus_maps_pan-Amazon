# -*- coding: utf-8 -*-
"""THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
   EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
   OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
   IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR
   ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
   TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
   SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from pickle import load
import numpy as np
from numba import jit

PFRACS = ["inorg_p", "org_p", "avail_p", "occ_p" , "total_p"]


@jit(nopython=True)
def find_coord(N:float,
               W:float,
               RES:float,
               lat:np.array=None,
               lon:np.array=None
               ) -> tuple[int, int]:
    """
    :param N:float: latitude in decimal degrees
    :param W:float: Longitude in decimal degrees
    :param RES:float: Resolution in degrees / must conform to the below data
    :param lat:numpy.array or (anything that a enumerate can eat) with the latitudes (center of pixels)
    :param lon:numpy.array or (anything that a enumerate can eat) with the longitudes (center of pixels)

    """

    Yc = round(N, 5)
    Xc = round(W, 5)

    for Yind, y in enumerate(lat):
        if abs(Yc - y) < RES/2:
           break

    for Xind, x in enumerate(lon):
        if abs(Xc - x)  < RES/2:
           break

    return Yind, Xind


def percentile_treshold(arr, percentile=75, norm=False):
    a = arr.flatten()
    b = a[a != -9999.0]
    if norm:
        b = np.sqrt(b)
    q1 = np.percentile(b, 25)
    pct = np.percentile(b, percentile)
    iqr = pct - q1
    return pct + iqr


def make_all_mask(all_arr):
    mask = all_arr[0] == -128
    all = np.array(np.logical_or.reduce(all_arr), dtype=np.int8)
    all[mask.mask] = -128
    return all


def make_di_mask(di):
    nodata_mask = di == -9999.0
    di_masked = np.ma.masked_array(di, nodata_mask, fill_value=-9999.0)
    bool_mask = di_masked >= percentile_treshold(di_masked.data)
    mask = np.array(bool_mask.data, dtype=np.bool_)
    mask = np.array(mask, dtype=np.int8)
    mask[nodata_mask] = -128
    return mask


def best_model(label):
    best_model = 0.0
    with open(f"./selected_models/models_{label}.pkl", "rb") as fh:
        models = load(fh)
    for i in range(len(models)):
        if models[i][4] > best_model:
            best_model = models[i][4]
            rf = models[i][2]
            idx = i
    print("Accuracy of model {}: {}".format(rf.random_state, best_model))
    return models[idx]


def best_30(label):
    with open(f"./selected_models/models_{label}.pkl", "rb") as fh:
        models = load(fh)
    # Sort models according to cross validation scores
    sorted_lst = sorted(models, key=lambda x:x[4], reverse=True)
    return sorted_lst[:30]

