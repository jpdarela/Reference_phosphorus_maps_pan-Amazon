import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances

from table2raster import col2arr

FILE = 6

# Calculate the Dissimilarity Index according to:
"""
Meyer, H. and Pebesma, E.: Predicting into unknown space?
Estimating the area of applicability of spatial prediction models,
Methods Ecol Evol, 12, 1620-1633, 10.1111/2041-210x.13650, 2021.
"""

# Training dataset
sampled_dataset = pd.read_csv("./inputDATA/fitting_dataset_one_hot.csv")
predictive_dataset = pd.read_csv("./inputDATA/predictor_dataset.csv")

# Open permutation importances and average them
imp_avail_p = pd.read_csv("permutation_importances_avg_avail_p.csv").mean(axis=0)
imp_inorg_p = pd.read_csv("permutation_importances_avg_inorg_p.csv").mean(axis=0)
imp_org_p = pd.read_csv("permutation_importances_avg_org_p.csv").mean(axis=0)
imp_total_p = pd.read_csv("permutation_importances_avg_total_p.csv").mean(axis=0)

# Eliminate negative values from permutation importances (Mean Decrease in Accuracy)
imp_avail_p[imp_avail_p < 0] = 0.0
imp_inorg_p[imp_inorg_p < 0] = 0.0
imp_org_p[imp_org_p < 0] = 0.0
imp_total_p[imp_total_p < 0] = 0.0

# get mean & std of the training dataset
MEAN = sampled_dataset.drop("total_p", axis=1).mean(axis=0)
STD = sampled_dataset.drop("total_p", axis=1).std(axis=0)

#SCALE precictor dataset REF1 EQ. 1:
# X_ij_scaled = (X_ij - MEAN)/STD

out = ['id', 'nx', 'ny']
pred_vars = predictive_dataset.drop(out, axis=1)

assert np.all(pred_vars.columns == STD.keys()) and np.all(pred_vars.columns == MEAN.keys())

norm_pred = (pred_vars - MEAN) / STD
norm_obs = (sampled_dataset.drop("total_p", axis=1) - MEAN) / STD

# weight by mean importances
norm_pred_AP  = norm_pred * imp_avail_p
norm_pred_SMP = norm_pred * imp_inorg_p
norm_pred_OP  = norm_pred * imp_org_p
norm_pred_TP  = norm_pred * imp_total_p

norm_obs_AP  = norm_obs * imp_avail_p
norm_obs_SMP = norm_obs * imp_inorg_p
norm_obs_OP  = norm_obs * imp_org_p
norm_obs_TP  = norm_obs * imp_total_p

names_P = ("AP", "SMP", "OP", "TP")
norm_preds = (norm_pred_AP, norm_pred_SMP, norm_pred_OP, norm_pred_TP)
norm_obss = (norm_obs_AP, norm_obs_SMP, norm_obs_OP, norm_obs_TP)

names = list(norm_pred_AP.columns)

# dissMAT = [pairwise_distances(np.array(x), n_jobs=4) for x in norm_preds]
dissOBS = [pairwise_distances(np.array(x), n_jobs=4) for x in norm_obss]


def calc_DI(norm_obs_data, norm_pred_data, name):
    DI = []

    avrg_diss = pairwise_distances(np.array(norm_obs_data)).mean()

    for i in range(norm_pred_data.shape[0]):
        new_prediction_location = norm_pred_data.iloc[i]
        new_prediction_location.name = f"npl_{i}"
        df = pd.concat([norm_obs_data.T, new_prediction_location], axis=1).T
        diss = pairwise_distances(np.array(df), n_jobs=4)
        pt = np.argmin(diss[-1,:108])
        dk = diss[-1, pt]
        DI.append(dk / avrg_diss)

    dataset = {"nx": predictive_dataset.nx.__array__(),
               "ny": predictive_dataset.ny.__array__(),
               "DI": np.array(DI)}

    filename = f"DI_{name}.csv"
    pd.DataFrame(dataset).to_csv(filename, index=False)
    arr = col2arr(filename, 0, ("DI",))[0]
    plt.hist(DI, bins=40)
    plt.savefig(f"HIST_{name}.png")
    plt.close()
    return arr


dt_AP = calc_DI(norm_obs_AP, norm_pred_AP, "AP")
dt_SMP = calc_DI(norm_obs_SMP, norm_pred_SMP, "SMP")
dt_OP = calc_DI(norm_obs_OP, norm_pred_OP, "OP")
dt_TP = calc_DI(norm_obs_TP, norm_pred_TP, "TP")


# ESRI ascii files
header = "ncols 720\nnrows 360\nxllcorner -180\nyllcorner -90\ncellsize 0.5\nnodata_value -9999.0"
np.savetxt("AOA_AP.asc", dt_AP, fmt="%0.4f", header=header, comments="")
np.savetxt("AOA_SMP.asc", dt_SMP, fmt="%0.4f", header=header, comments="")
np.savetxt("AOA_OP.asc", dt_OP, fmt="%0.4f", header=header, comments="")
np.savetxt("AOA_TP.asc", dt_TP, fmt="%0.4f", header=header, comments="")

# save numpy arrays
np.save("AOA_AP.npy", dt_AP.data)
np.save("AOA_SMP.npy", dt_SMP.data)
np.save("AOA_OP.npy", dt_OP.data)
np.save("AOA_TP.npy", dt_TP.data)
