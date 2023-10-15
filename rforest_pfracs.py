# -*- coding: utf-8 -*-
"""THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
   EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
   OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
   IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR
   ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
   TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
   SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
from os import makedirs
from pathlib import Path

import sys
import pickle as pkl
import multiprocessing as mp
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

FILE = 1

NMODELS = 100000

"""FIT RANDOM FOREST (RF) REGRESSORS AND SELECT THE MOST ACCURATE MODELS
   USAGE: $ python rforest_pfracs.py <x>
   Where x is in ["inorg_p", "org_p", "avail_p", "total_p"].

   SIDE EFFECTS: create pickles with selected RF models
   """
# Dump pickled selected models here:
dump_folder = Path("./selected_models").resolve()
makedirs(dump_folder, exist_ok=True)

pforms = ['Inorganic P', 'Organic P',
          'Available P (Labile & Soluble)', 'Total P', "Occluded P", "Primary mineral P"]

SLICE = NMODELS // 10

n = str(sys.argv[1])

pfracs = ["inorg_p", "org_p", "avail_p", "total_p"] + ["occ_p", "mineral_p"]

assert n in pfracs, "Look the makefile"

label_name = n
lb_ = pforms[pfracs.index(n)]

if label_name == 'avail_p':
    cv_limit = 0.55
    SELECT_CRITERION = 75.0

elif label_name == 'inorg_p':
    cv_limit = 0.55
    SELECT_CRITERION = 65.0

elif label_name == 'org_p':
    cv_limit = 0.55
    SELECT_CRITERION = 73.0

elif label_name == "total_p":
    cv_limit = 0.55
    SELECT_CRITERION = 75.8

elif label_name == "occ_p":
    cv_limit = 0.50
    SELECT_CRITERION = 60

elif label_name == "mineral_p":
    cv_limit = 0.1
    SELECT_CRITERION = 5

features = pd.read_csv("./inputDATA/fitting_dataset.csv")

# # Choose the features
feat_used = ["lat", "lon", "RSG", "Sand", "Silt", "Clay",
             "Slope", "Elevation", "MAT", "MAP",
             "pH", "TOC", "TN", label_name]

clean_data = features[feat_used]

# # One-hot encoding for nominal variables ('RSG')
# dta = pd.get_dummies(clean_data, dtype=float)
dta = pd.get_dummies(clean_data, dtype=float, prefix="", prefix_sep="")

# Variable to be predicted as an np.array
labels = np.array(dta[label_name])

# Exclude labels from features
feat = dta.drop(label_name, axis=1)

# Get the features names
feat_list = list(feat.columns)

# Transform in a NP.array
feat = np.array(feat)


def check_crossval(arr, limit):
    return arr.mean() >= limit

# Helper to multiprocessing
def chunks(lst, chunck_size):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), chunck_size):
        yield lst[i:i + chunck_size]

# FIT MODELS AND ESTIMATE ACCURACY (via MAPE)
def make_model(index):
    cv = ShuffleSplit(test_size=0.25, n_splits=15, random_state=index)

    train_features, test_features, \
        train_labels, test_labels = \
            train_test_split(feat, labels, test_size=0.25,
                             random_state=index)

    # Create the random forest
    rf = RandomForestRegressor(n_estimators=100, criterion="squared_error",
                               n_jobs=1, random_state=index)

    # FIT
    rf.fit(train_features, train_labels)

    scores = cross_val_score(rf, feat, labels, cv=cv, scoring="r2", n_jobs=1)
    # print(scores)

    # Use the forest's predict method on the test data
    predictions = rf.predict(test_features)

    # Calculate the absolute errors
    errors = mean_absolute_error(test_labels, predictions)

    # Calculate mean absolute percentage error (MAPE)
    mape = mean_absolute_percentage_error(test_labels, predictions) * 100.0

    # Calculate accuracy
    accuracy = 100.0 - mape

    if accuracy >= SELECT_CRITERION and check_crossval(scores, cv_limit):
        r2 = r2_score(test_labels, predictions)
        print("MODEL random state:", index)
        print('Mean Absolute Error:', round(np.mean(errors), 2), 'mg/kg')
        print('Accuracy:', round(accuracy, 2), '%')
        print("R2: ", r2)
        print("Score CV: ", scores.mean())
        print(" ")
        return (index, accuracy, rf, r2, scores.mean(), scores.std(), np.mean(errors))
    else: return None


if __name__ == "__main__":
    # FIT & SAVE selected models to a pickle
    result = []
    for lst in chunks(np.arange(1,NMODELS + 1), SLICE):
        with mp.Pool(processes=mp.cpu_count() - 1) as pool:
            result += pool.map(make_model, lst)

    models = [a for a in result if a is not None]
    fname = Path(f"models_{label_name}.pkl")
    with open(dump_folder/fname, mode="wb") as fh:
        pkl.dump(models, fh)
