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

from numba import jit
from numpy import array
from pandas import read_csv, get_dummies

PFRACS = ["occ_p"]#["inorg_p", "org_p", "avail_p", "total_p", "occ_p", "mineral_p"]

@jit(nopython=True)
def find_coord(N:float, W:float, RES:float, lat, lon) -> tuple[int, int]:
    """

    :param N:float: latitude in decimal degrees
    :param W:float: Longitude in decimal degrees
    :param RES:float: Resolution in degrees (Default value = 0.5)

    """

    Yc = round(N, 5)
    Xc = round(W, 5)

    for Yind, y in enumerate(lat):
        if abs(Yc - y) < RES/4:
           break

    for Xind, x in enumerate(lon):
        if abs(Xc - x)  < RES/4:
           break

    return Yind, Xind


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


def get_dataset(label_name, with_names=False):
    features = read_csv("./inputDATA/fitting_dataset.csv")

# # Choose the features
    feat_used = ["lat", "lon", "SRG", "Sand", "Silt", "Clay",
                "Slope", "Elevation", "MAT", "MAP",
                "pH", "TOC", "TN", label_name]

    clean_data = features[feat_used]

    # # One-hot encoding for nominal variables ('SRG')
    dta = get_dummies(clean_data)

    # Variable to be predicted as an np.array
    label = array(dta[label_name])

    # Exclude labels from features
    feat = dta.drop(label_name, axis=1)

    # Get the features names
    feat_list = list(feat.columns)

    # Transform in a NP.array
    features = array(feat)
    if with_names:
        return features, feat_list, label, label_name
    return features, label
