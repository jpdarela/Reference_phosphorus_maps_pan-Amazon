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
import pickle as pkl
import pandas as pd
from numpy import mean


FILE = 2

"""Apply the selected models to predict P forms over the Study Area (Predict with spatially 
   explicit data. Also generates tables with model evaluation metrics and metadata)
   USAGE: $ python predict_P.py
   SIDE EFFECTS: .csv tables with predicted values for each selected model for each P form"""


feat_list = ["lat","lon","Sand","Silt","Clay","Slope","Elevation","MAT","MAP","pH","TOC",
    "TN","SRG_Acrisols","SRG_Alisols","SRG_Andosols","SRG_Arenosols","SRG_Cambisols",
    "SRG_Ferralsols","SRG_Fluvisols","SRG_Gleysols","SRG_Lixisols","SRG_Luvisols","SRG_Nitisols",
    "SRG_Plinthosols","SRG_Podzol","SRG_Regosols","SRG_Umbrisols"]

pfracs = ["inorg_p", "org_p", "avail_p", "total_p"]


fh = open("model_selection_scores.csv", 'w')
fh.write(",".join(['pform', 'nmodels', 'mean_acc', "mae", "r2", "cv_mean", "cv_std" + "\n"]))
fh.close()

for label_name in pfracs:

    output = "./predicted_P_%s/" % label_name
    model = "models_"

    predictors = './inputDATA/predictor_dataset.csv'

    if not os.path.exists(output): os.mkdir(output) 

    with open("%s%s.pkl" %(model, label_name), 'rb') as fh:
        models = pkl.load(fh)

    acc = []
    mae = []
    r2 = []
    cv_mean = []
    cv_std = []
    r_state = []

    for i, md in enumerate(models):
        
        # Predictors table
        map_data = pd.read_csv(predictors)[feat_list].__array__()

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

        map_data = pd.read_csv(predictors)

        map_data[label_name] = new_column
        map_data.to_csv(output + "/predicted_%s_model_%s.csv" % (label_name, str(md[0])), index=False, header=True)

    with open('model_selection_scores.csv', 'a') as csv_file:
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
    df.to_csv(f"eval_metrics_{label_name}.csv", index=False)

