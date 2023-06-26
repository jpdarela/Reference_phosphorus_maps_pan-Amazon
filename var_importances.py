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
import csv
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
import pickle as pkl
import pandas as pd
import numpy as np

FILE = 5

""" Extract/Calculate the Gini/permutation importances of the selected RF models  """

print("Calculating variables importances...")
files = glob1(os.getcwd(), "models*")
varns = ["_".join(fl.split(".")[0].split('_')[1:]) for fl in files]

feat_list = ["lat","lon","Sand","Silt","Clay","Slope","Elevation","MAT","MAP","pH","TOC",
    "TN","SRG_Acrisols","SRG_Alisols","SRG_Andosols","SRG_Arenosols","SRG_Cambisols",
    "SRG_Ferralsols","SRG_Fluvisols","SRG_Gleysols","SRG_Lixisols","SRG_Luvisols","SRG_Nitisols",
    "SRG_Plinthosols","SRG_Podzol","SRG_Regosols","SRG_Umbrisols"]


for fh, varn in zip(files, varns):

    with open(fh, "rb") as fhand:
        models = pkl.load(fhand)

    with open("importances_%s.csv" % varn, mode="w") as csvfile:
        fwriter = csv.DictWriter(csvfile, fieldnames=feat_list)
        fwriter.writeheader()

        for model in models:
            rf = model[2]
            importances = list(rf.feature_importances_)

            # List of tuples with variable and importance
            feature_importances = [(feat, importance) for feat, importance in zip(feat_list, importances)]
            # print(dict(feature_importances))
            fwriter.writerow(dict(feature_importances))
 
        # Permutation importances
        label_name = varn
        features = pd.read_csv("./inputDATA/fitting_dataset.csv")

        # # Choose the important _features 
        feat_used = ["lat", "lon", "SRG", "Sand", "Silt", "Clay", "Slope", "Elevation", "MAT", "MAP",
                    "pH", "TOC", "TN", label_name]

        clean_data = features[feat_used]

        # # One-hot encoding for nominal variables ('s_type2')
        dta = pd.get_dummies(clean_data)

        # Variable to be predicted as an np.array
        labels = np.array(dta[label_name])

        # Exclude labels from features
        feat = dta.drop(label_name, axis=1)

        # Get the features names
        feat_list = list(feat.columns)

        # Transform in a NP.array
        feat = np.array(feat)

    with open("permutation_importances_avg_%s.csv" % varn, mode="w") as csvfile1,\
         open("permutation_importances_std_%s.csv" % varn, mode="w") as csvfile2:
        
        fwriter1 = csv.DictWriter(csvfile1, fieldnames=feat_list, dialect="unix")
        fwriter2 = csv.DictWriter(csvfile2, fieldnames=feat_list, dialect="unix")
        fwriter1.writeheader()
        fwriter2.writeheader()

        for model in models:
            rf = model[2]
            random_state = model[0]
            train_features, test_features,\
                train_labels, test_labels = \
                    train_test_split(feat, labels, test_size=0.25,
                                     random_state=random_state)
            
            importances = permutation_importance(rf, test_features, test_labels,
                                     n_repeats=120, n_jobs=28)

            # List of tuples with variable and importance
            feature_importances_avg = [(feat, importance) for feat, importance in zip(feat_list, importances.importances_mean)]
            # print(dict(feature_importances_avg))
            fwriter1.writerow(dict(feature_importances_avg))
            
            feature_importances_std = [(feat, importance) for feat, importance in zip(feat_list, importances.importances_std)]
            # print(dict(feature_importances_std))
            fwriter2.writerow(dict(feature_importances_std))
