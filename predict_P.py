# -*- coding: utf-8 -*-
"""THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
   EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
   OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
   IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR
   ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
   TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
   SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
from copy import deepcopy
from pathlib import Path
import os
import pickle as pkl
import pandas as pd
from numpy import mean
from utilities import PFRACS


FILE = 2

"""Apply the selected models to predict P forms over the Study Area (Predict with spatially
   explicit data. Also generates tables with model evaluation metrics and metadata)
   USAGE: $ python predict_P.py
   SIDE EFFECTS: .csv tables with predicted values for each selected model for each P form"""


feat_list = ["lat","lon","Sand","Silt","Clay","Slope","Elevation","MAT","MAP","pH","TOC",
    "TN","Acrisols","Alisols","Andosols","Arenosols","Cambisols",
    "Ferralsols","Fluvisols","Gleysols","Lixisols","Luvisols","Nitisols",
    "Plinthosols","Podzols","Regosols","Umbrisols"]

pfracs = PFRACS

models = Path("./selected_models/")

metrics = Path("./model_evaluation_metrics")
os.makedirs(metrics, exist_ok=True)

fh = open(metrics/Path("model_selection_scores.csv"), 'w')
fh.write(",".join(['pform', 'nmodels', 'mean_acc', "mae", "r2", "cv_mean", "cv_std" + "\n"]))
fh.close()

predictors = './inputDATA/predictive.csv'

predictive_dataset = pd.read_csv(predictors)

map_data = predictive_dataset[feat_list].__array__()
template = predictive_dataset[["OID_","lat", "lon"]]

for label_name in pfracs:

    output = Path(f"predicted_P_{label_name}").resolve()
    os.makedirs(output, exist_ok=True)
    model = "models_"

    with open(models/Path("%s%s.pkl" %(model, label_name)), 'rb') as fh:
        selected_models = pkl.load(fh)

    acc = []
    mae = []
    r2 = []
    cv_mean = []
    cv_std = []
    r_state = []

    for i, md in enumerate(selected_models):

        # Get the model
        rf = md[2]

        r_state.append(md[0])
        acc.append(md[1])
        r2.append(md[3])
        cv_mean.append(md[4])
        cv_std.append(md[5])
        mae.append(md[6])

        new_column = rf.predict(map_data)

        if len(new_column.shape) > 1:
            new_column = new_column[:,0]

        output_data = deepcopy(template)

        output_data[label_name] = new_column
        output_data.to_feather(output/Path("predicted_%s_model_%s.feather" % (label_name, str(md[0]))))

    with open(metrics/Path('model_selection_scores.csv'), 'a') as csv_file:
        line = [label_name, str(i+1), str(round(sum(acc) / (i + 1), 2)), str(mean(mae)), str(mean(r2)), str(mean(cv_mean)), str(mean(cv_std)) + "\n"]
        csv_file.write(",".join(line))
        print(label_name, 'number of models: ', i + 1, end='-> ')
        print('Mean acc: ', sum(acc) / (i + 1))

    index = list(map(int, range(len(r_state))))
    eval_metrics= {"random_state" : r_state,
                   "accuracy": acc,
                   "R2": r2,
                   "CV_mean": cv_mean,
                   "CV_std": cv_std,
                   "MAE": mae}

    df = pd.DataFrame(data=eval_metrics, index=index)
    df.to_csv(metrics/Path(f"eval_metrics_{label_name}.csv"), index=False)

