# -*- coding: utf-8 -*-
"""THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
   EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
   OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
   IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR
   ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
   TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
   SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances

from table2raster import rasterize, ydim, xdim, cell_size, lat, lon
from utilities import make_di_mask, make_all_mask

FILE = 5

# Calculate the Dissimilarity Index according to:
"""
Meyer, H. and Pebesma, E.: Predicting into unknown space?
Estimating the area of applicability of spatial prediction models,
Methods Ecol Evol, 12, 1620-1633, 10.1111/2041-210x.13650, 2021.
"""

# Training dataset
sampled_dataset = pd.read_csv("./inputDATA/fitting_dataset_one_hot.csv").drop("id", axis=1)
predictive_dataset = pd.read_csv("./inputDATA/predictive.csv").drop(['OID_', 'pointid'], axis=1)

# Open permutation importances and average them. We use mean permutation of all models
imp_avail_p = pd.read_csv("./MDA/permutation_importances_avg_avail_p.csv").mean(axis=0)
imp_inorg_p = pd.read_csv("./MDA/permutation_importances_avg_inorg_p.csv").mean(axis=0)
imp_org_p = pd.read_csv("./MDA/permutation_importances_avg_org_p.csv").mean(axis=0)
imp_occ_p = pd.read_csv("./MDA/permutation_importances_avg_occ_p.csv").mean(axis=0)
imp_total_p = pd.read_csv("./MDA/permutation_importances_avg_total_p.csv").mean(axis=0)

# Eliminate negative values from permutation importances (Mean Decrease in Accuracy)
imp_avail_p[imp_avail_p < 0] = 0.0
imp_inorg_p[imp_inorg_p < 0] = 0.0
imp_org_p[imp_org_p < 0] = 0.0
imp_occ_p[imp_occ_p < 0] = 0.0
imp_total_p[imp_total_p < 0] = 0.0

# get mean & std of the training dataset
MEAN = sampled_dataset.mean(axis=0)
STD = sampled_dataset.std(axis=0)

#SCALE precictor dataset REF1 EQ. 1:
# X_ij_scaled = (X_ij - MEAN)/STD

assert np.all(predictive_dataset.columns == STD.keys()) and np.all(predictive_dataset.columns == MEAN.keys())

norm_pred = (predictive_dataset - MEAN) / STD
norm_obs = (sampled_dataset - MEAN) / STD

# weight by mean importances
norm_pred_AP  = norm_pred * imp_avail_p
norm_pred_SMP = norm_pred * imp_inorg_p
norm_pred_OP  = norm_pred * imp_org_p
norm_pred_OCP  = norm_pred * imp_occ_p
norm_pred_TP  = norm_pred * imp_total_p

norm_obs_AP  = norm_obs * imp_avail_p
norm_obs_SMP = norm_obs * imp_inorg_p
norm_obs_OP  = norm_obs * imp_org_p
norm_obs_OCP  = norm_obs * imp_occ_p
norm_obs_TP  = norm_obs * imp_total_p

names_P = ("AP", "SMP", "OP", "OCP", "TP")
norm_obss = (norm_obs_AP, norm_obs_SMP, norm_obs_OP, norm_obs_OCP, norm_obs_TP)

names = list(norm_pred_AP.columns)

dissOBS = [pairwise_distances(np.array(x)) for x in norm_obss]


def calc_DI(norm_obs_data, norm_pred_data, name):
    DI = []

    avrg_diss = pairwise_distances(np.array(norm_obs_data)).mean()

    observed_dataset = norm_obs_data.__array__()

    y, x = observed_dataset.shape

    tmp = np.zeros(shape=(y + 1, x))
    tmp[:108, :] = observed_dataset

    pred1 = predictive_dataset.iloc(0)
    c = 0
    for row in norm_pred_data.iloc(0):

        tmp[-1, :] = row.__array__()
        diss = pairwise_distances(tmp)
        # return diss, tmp
        pt = np.argmin(diss[-1,:108])
        dk = diss[-1, pt]
        DI.append((pred1[c].lat, pred1[c].lon, dk / avrg_diss))
        print(f"\rTesting row {c} of the predictive dataset for {name}\t\t", end="", flush=True)
        c += 1
        # if c > 100000:
        #     break
    DI = np.array(DI)
    img = rasterize(DI, clean=False, offset=1)
    print("OK")
    return img


dt_AP = calc_DI(norm_obs_AP, norm_pred_AP, "AP")
dt_SMP = calc_DI(norm_obs_SMP, norm_pred_SMP, "SMP")
dt_OP = calc_DI(norm_obs_OP, norm_pred_OP, "OP")
dt_OCP = calc_DI(norm_obs_OCP, norm_pred_OCP, "OCP")
dt_TP = calc_DI(norm_obs_TP, norm_pred_TP, "TP")


# # ESRI ascii files
xllcorner = lon[0] - cell_size / 2
yllcorner = lat[-1] - cell_size / 2

header = f"ncols {xdim}\nnrows {ydim}\nxllcorner {xllcorner}\nyllcorner {yllcorner}\ncellsize {cell_size}\nnodata_value -9999.0"

np.savetxt("./dissimilarity_index_masks/DI_avail_p.asc", dt_AP, fmt="%0.4f", header=header, comments="")
np.savetxt("./dissimilarity_index_masks/DI_inorg_p.asc", dt_SMP, fmt="%0.4f", header=header, comments="")
np.savetxt("./dissimilarity_index_masks/DI_org_p.asc", dt_OP, fmt="%0.4f", header=header, comments="")
np.savetxt("./dissimilarity_index_masks/DI_occ_p.asc", dt_OCP, fmt="%0.4f", header=header, comments="")
np.savetxt("./dissimilarity_index_masks/DI_total_p.asc", dt_TP, fmt="%0.4f", header=header, comments="")


header = f"ncols {xdim}\nnrows {ydim}\nxllcorner {xllcorner}\nyllcorner {yllcorner}\ncellsize {cell_size}\nnodata_value -128"
no_data = -128

mask_ap = make_di_mask(dt_AP)
np.savetxt("./dissimilarity_index_masks/DI_mask_avail_p.asc", mask_ap, fmt="%d",header=header, comments="")
np.save("./dissimilarity_index_masks/DI_mask_avail_p.npy", mask_ap)

mask_ip = make_di_mask(dt_SMP)
np.savetxt("./dissimilarity_index_masks/DI_mask_inorg_p.asc", mask_ip, fmt="%d",header=header, comments="")
np.save("./dissimilarity_index_masks/DI_mask_inorg_p.npy", mask_ip)

mask_op = make_di_mask(dt_OP)
np.savetxt("./dissimilarity_index_masks/DI_mask_org_p.asc", mask_op, fmt="%d",header=header, comments="")
np.save("./dissimilarity_index_masks/DI_mask_org_p.npy", mask_op)

mask_ocp = make_di_mask(dt_OCP)
np.savetxt("./dissimilarity_index_masks/DI_mask_occ_p.asc", mask_ocp, fmt="%d",header=header, comments="")
np.save("./dissimilarity_index_masks/DI_mask_occ_p.npy", mask_ocp)

mask_tp = make_di_mask(dt_TP)
np.savetxt("./dissimilarity_index_masks/DI_mask_total_p.asc", mask_tp, fmt="%d",header=header, comments="")
np.save("./dissimilarity_index_masks/DI_mask_total_p.npy", mask_tp)

ma = np.ma.masked_array

all_arr = [ma(mask_tp, mask=mask_tp==no_data),
           ma(mask_op, mask=mask_op==no_data),
           ma(mask_ap, mask=mask_ap==no_data),
           ma(mask_ip, mask=mask_ip==no_data),
           ma(mask_ocp, mask=mask_ocp==no_data),
           ]

all = make_all_mask(all_arr)
# all[mask2] = no_data
np.savetxt("./dissimilarity_index_masks/DI_mask_ALL.asc", all, fmt="%d",header=header, comments="")
np.save("./dissimilarity_index_masks/DI_mask_ALL.npy", all)
