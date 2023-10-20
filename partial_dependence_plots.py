# -*- coding: utf-8 -*-
"""THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
   EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
   OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
   IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR
   ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
   TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
   SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from pathlib import Path
from os import makedirs, path, getcwd
from pickle import load

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d  # noqa: F401

from sklearn.inspection import PartialDependenceDisplay, partial_dependence
from sklearn.model_selection import train_test_split

pforms = ['Inorganic P', 'Organic P',
          'Available P', 'Total P', 'Occluded P']
pfracs = ["inorg_p", "org_p", "avail_p", "total_p", "occ_p"]

names = dict(zip(pfracs, pforms))

features = pd.read_csv("./inputDATA/fitting_dataset.csv")

pathin = path.join(Path(getcwd()), Path("PDP_PLOTS"))

makedirs(pathin, exist_ok=True)

def feat_to_get(label):
    return ["lat", "lon", "RSG","Sand", "Silt", "Clay",
            "Slope", "Elevation", "MAT", "MAP",
            "pH", "TOC", "TN", label]

def most_important(imp, nloop=5):
    feat = list(imp.keys())
    importances = imp.__array__()
    features = []
    for x in range(nloop):
        features.append((feat[np.argmax(importances)], np.argmax(importances)))
        importances[np.argmax(importances)] = -1e5
    return features

def best_model(label):
    best_model = 0.0
    with open(f"./models_{label}.pkl", "rb") as fh:
        models = load(fh)
    for i in range(len(models)):
        if models[i][1] > best_model:
            best_model = models[i][1]
            rf = models[i]
    print("Accuracy of model {}: {}".format(rf[2].random_state, best_model))
    return rf

def worst_model(label):
    worst_model = 1e15
    with open(f"./models_{label}.pkl", "rb") as fh:
        models = load(fh)
    for i in range(len(models)):
        if models[i][1] < worst_model:
            worst_model = models[i][1]
            rf = models[i]
    print("Accuracy of model {}: {}".format(rf[2].random_state, worst_model))
    return rf

def get_importances(label):
    return pd.read_csv(f"permutation_importances_avg_{label}.csv").mean()

def perm_imp(label, threeD=False, v1=0, v2=1):
    imp = get_importances(label)
    features = most_important(imp)
    if not threeD:
        return {"features": [features[v1][0], features[v2][0], (features[v1][0], features[v2][0])],
                "kind": "average"}
    return (features[v1][0], features[v2][0])

def plotPDP2W(label):#, features_info, vars):
    model = best_model(label)

    data = features[feat_to_get(label)]

    data = pd.get_dummies(data)

    labels = np.array(data[label])

    # Exclude labels from features
    feat = data.drop(label, axis=1)

    feat_list = list(feat.columns)

    rf = model[2]
    state = model[0]
    print(state)

    train_features, test_features, \
        train_labels, test_labels = \
            train_test_split(feat, labels, test_size = 0.25,
                                random_state = state)

    features_info = perm_imp(label)
    fig, ax = plt.subplots(ncols=3, figsize=(10, 4), constrained_layout=True)
    display = PartialDependenceDisplay.from_estimator(
        rf,
        train_features,
        **features_info,
        ax=ax,
    )
    _ = display.figure_.suptitle(
        "Partial dependency", fontsize=16
    )
    plt.savefig(path.join(pathin, f"PDP_plots_2W_{state}-{label}.png"), dpi=300)
    plt.close(fig)
    return None

def plotPDP3d(label, v1, v2, best=True):#, features_info, vars):
    if best:
        model = best_model(label)
    else:
        model = worst_model(label)

    data = features[feat_to_get(label)]

    data = pd.get_dummies(data)

    labels = np.array(data[label])

    # Exclude labels from features
    feat = data.drop(label, axis=1)

    feat_list = list(feat.columns)

    rf = model[2]
    state = model[0]
    print(state)

    train_features, test_features, \
        train_labels, test_labels = \
            train_test_split(feat, labels, test_size = 0.25,
                                random_state = state)

    features_info = []
    f1 = perm_imp(label, threeD=True, v1=v1, v2=v2)
    features_info.append(f1[0])
    features_info.append(f1[1])
    pdp = partial_dependence(rf, train_features, features=features_info,
                              kind="average", grid_resolution=20)

    fig = plt.figure(figsize=(5.5, 5))
    XX, YY = np.meshgrid(pdp["values"][0], pdp["values"][1])
    Z = pdp.average[0].T
    ax = fig.add_subplot(projection="3d")
    fig.add_axes(ax)

    surf = ax.plot_surface(XX, YY, Z, rstride=1, cstride=1, cmap=plt.cm.Blues, edgecolor="k")
    ax.set_xlabel(features_info[0])
    ax.set_ylabel(features_info[1])
    fig.suptitle(
        f"PD of {names[label]} on the variables {features_info[0]} and {features_info[1]}",
        fontsize=10,
    )
    # pretty init view
    ax.view_init(elev=22, azim=122)
    clb = plt.colorbar(surf, pad=0.08, shrink=0.6, aspect=10)
    clb.ax.set_title("Partial\ndependence")
    plt.savefig(path.join(pathin, f"PDP_plots_3D_{state}-{label}.png"), dpi=300)
    plt.close(fig)
    return pdp

def calcPD(label):#, features_info, vars):

    model = best_model(label)

    data = features[feat_to_get(label)]

    data = pd.get_dummies(data)

    labels = np.array(data[label])

    # Exclude labels from features
    feat = data.drop(label, axis=1)

    feat_list = list(feat.columns)

    rf = model[2]
    state = model[0]
    print(state)

    train_features, test_features, \
        train_labels, test_labels = \
            train_test_split(feat, labels, test_size = 0.25,
                                random_state = state)

    imp = get_importances(label)
    features_info = [x[0] for x in most_important(imp=imp, nloop=9)]
    pdp = []
    for f in features_info:
        pdp.append(partial_dependence(rf, train_features, features=(f,),
                                      kind="average", grid_resolution=100))

    return pdp

def plotPD(label, best=True):#, features_info, vars):
    if best:
        model = best_model(label)
        s1 = "best"
    else:
        model = worst_model(label)
        s1 = "worse"


    data = features[feat_to_get(label)]

    data = pd.get_dummies(data)

    labels = np.array(data[label])

    # Exclude labels from features
    feat = data.drop(label, axis=1)

    feat_list = list(feat.columns)

    rf = model[2]
    state = model[0]
    print(state)

    train_features, test_features, \
        train_labels, test_labels = \
            train_test_split(feat, labels, test_size = 0.25,
                                random_state = state)

    imp = get_importances(label)
    features_info = {"features": [x[0] for x in most_important(imp=imp, nloop=2)],
                     "kind": "both"}
    fig, ax = plt.subplots(nrows=1, ncols=2, constrained_layout=True)

    display = PartialDependenceDisplay.from_estimator(
        rf,
        train_features,
        **features_info,
        ax=ax)
    display.figure_.suptitle(f"Partial dependence of {names[label]}", fontsize=12)
    plt.savefig(f"./p_figs/ICE_plot_{label}_{s1}_2_.png", dpi=300)
    plt.close(fig)

def plotICE(label="total_p"):
    pass

if __name__ == "__main__":
    for v in pfracs:
        plotPD(v)
        plotPDP2W(v)
        plotPDP3d(v, 0, 1)