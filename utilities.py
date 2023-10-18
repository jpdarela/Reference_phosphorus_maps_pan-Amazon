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
    # TODO normalize first?
    q1 = np.percentile(b, 25)
    pct = np.percentile(b, percentile)
    iqr = pct - q1
    # print("3rd quartile + IQR: ", pct + iqr)
    return pct + iqr

# def make_all_mask():
#     mask = []
#     for pfrac in var:
#         mask.append(make_DI_mask(pfrac))
#     internal_mask = mask[0].mask
#     return np.ma.masked_array(np.logical_or.reduce(mask), mask=internal_mask)

def make_di_mask(di):
    di_masked = np.ma.masked_array(di, di == -9999.0, fill_value=-9999.0)
    return np.logical_not(np.array((di_masked >= percentile_treshold(di_masked.data)).data, dtype=np.bool_))


def best_model(label):
    best_model = 0.0
    with open(f"./models_{label}.pkl", "rb") as fh:
        models = load(fh)
    for i in range(len(models)):
        if models[i][4] > best_model:
            best_model = models[i][4]
            rf = models[i][2]
    print("Accuracy of model {}: {}".format(rf.random_state, best_model))
    return rf.random_state
