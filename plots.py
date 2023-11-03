# -*- coding: utf-8 -*-
"""THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
   EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
   OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
   IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR
   ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
   TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
   SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import csv
import os

from cartopy.feature import ShapelyFeature
from cartopy.io.shapereader import Reader
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib.lines import Line2D
from matplotlib.path import Path
from netCDF4 import Dataset
from sklearn.inspection import PartialDependenceDisplay
from sklearn.model_selection import train_test_split

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cfunits
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from scientific_colourmaps import batlowK_map as batlowK
from table2raster import cell_size, lat, lon, area
from table2raster import mask as raster_mask
from utilities import best_model

FILE = 7

pa_area = np.ma.masked_array(area, mask = raster_mask)

EXTENT = [lon[ 0] - cell_size / 2,
          lon[-1] + cell_size / 2,
          lat[-1] - cell_size / 2,
          lat[ 0] + cell_size / 2]

FOCUS_EXTENT = [-80, -43.5, -21, 11]

os.makedirs("p_figs", exist_ok=True)

legend_elements = [Line2D([0], [0], color='k', lw=1, label='Pan-Amazon'),]

pforms = ['Inorganic', 'Organic', 'Available',
          'Total', 'Occluded', 'Primary']

var = ['inorg_p', 'org_p', 'avail_p', 'total_p', 'occ_p', 'mineral_p']

labels = dict(list(zip(var, pforms)))

NOT_SOIL_FEATURES = ["lat", "lon", "Sand", "Silt", "Clay", "Slope",
                     "Elevation", "MAT", "MAP", "pH", "TOC", "TN"]

SOIL_FEATURES = ["Acrisols", "Alisols", "Andosols", "Arenosols",
                 "Cambisols", "Ferralsols", "Fluvisols", "Gleysols",
                 "Lixisols", "Luvisols", "Nitisols", "Plinthosols",
                 "Podzols", "Regosols", "Umbrisols"]

mask = ShapelyFeature(Reader("./inputDATA/shp/pan_Amazon_mask.shp").geometries(),
                                 ccrs.PlateCarree())

scores = pd.read_csv('./model_evaluation_metrics/model_selection_scores.csv', index_col='pform')


def get_DI_mask(pfrac):
    array = np.load(f"./dissimilarity_index_masks/DI_mask_{pfrac}.npy")
    no_data_mask = array == -128
    di_mask = array == 1
    return np.logical_or.reduce([no_data_mask, di_mask])


def plot_Pmap(vname):
    """PLOT P mean maps in mg kg-1 + fraction of total p + SE + scores"""
    plt.rc('legend', fontsize=4.8)
    plt.rc('font', size=6) #controls default text size
    plt.rc('axes', titlesize=6) #fontsize of the title
    plt.rc('xtick', labelsize=6) #fontsize of the x tick labels
    plt.rc('ytick', labelsize=6) #fontsize of the y tick labels

    units = "mg kg⁻¹"

    # Create a marker
    theta = np.linspace(0, 2 * np.pi, 100)
    circle_verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    concentric_circle = Path.make_compound_path(Path(circle_verts[::-1]),
                                                Path(circle_verts * 1))
    # read points
    points = []
    with open('./inputDATA/fitting_dataset.csv', 'r') as fh:
        reader = csv.reader(fh)
        for line in reader:
            points.append(line[2:4])
    legend_elements = [Line2D([0], [0], color='k', lw=1, label='Pan-Amazon'),]
    pt = np.array(points[1:][:], dtype=np.float32)


    title = labels[vname]
    if vname != "mineral_p":
        DI_mask = get_DI_mask(vname)
        nmodels = f"n {scores.loc[[vname]]['nmodels'].get(vname)}"
        accuracy = round(scores.loc[[vname]]['mean_acc'].get(vname), 1)
        Amean = f"Aµ {accuracy} %"
    else:
        DI_mask = get_DI_mask("ALL")

    if vname == 'total_p':
        frac_of_total = None

        with Dataset("./results/total_p_AVG.nc", "r") as fh:
            img = fh.variables[vname][:]

    elif vname == 'mineral_p':
        with Dataset("./results/mineral_p.nc", "r") as fh:
            mineral = fh.variables[vname][:]
        with Dataset("./results/mineral_p_percentage_of_total_p.nc", "r") as fh:
            frac_of_total = fh.variables[vname][:]
        # mask_di = get_DI_mask("ALL")
        nmodels = ""
        Amean = ""
        idx = np.where(mineral < 0)
        mineral[idx] = 0.0
        frac_of_total [idx] = 0.0
        # frac_of_total = np.ma.masked_array(frac_of_total, mask=mask_di)
        img = mineral#!np.ma.masked_array(mineral, mask=mask_di)

    else:
        with Dataset(f"./results/{vname}_percentage_of_total_p.nc", "r") as fh:
            frac_of_total = fh.variables[vname][:]
        with Dataset(f"./results/{vname}_AVG.nc", "r") as fh:
            img = fh.variables[vname][:]

    if not vname == "mineral_p":
        with Dataset(f"./results/{vname}_SE.nc", "r") as fh:
            SE = fh.variables[vname][:]

    ## ---------PLOT MAP-----------------
    # Raster extent
    img_proj = ccrs.PlateCarree()
    img_extent = EXTENT

    if vname == 'total_p' or vname == "mineral_p":
        fig = plt.figure(figsize=(0.66 * 8.2, 2.6))
        gs = gridspec.GridSpec(1, 2)
        srk= 0.704
        if vname == "mineral_p":
            srk= 0.73
        _x_= 32.5
        _y_= 33
    else:
        fig = plt.figure(figsize=(8.2, 2.6))
        gs = gridspec.GridSpec(1, 3)
        srk = 0.695# 0.763 # For org_p # AVAIL and INORG = 0.0.765
        _x_= 43
        _y_= 33

    # Main Map
    ax = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
    ax.set_xticks([-70, -45], crs=ccrs.PlateCarree())
    ax.set_yticks([-16, 10], crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=True, number_format='g')
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)

    ax.set_title(title, fontsize='medium')
    ax.set_extent(FOCUS_EXTENT, crs=ccrs.PlateCarree())

    imsh = plt.imshow(np.ma.masked_array(img, mask=DI_mask), transform=img_proj, extent=img_extent, cmap=batlowK, zorder=3)
    ax.add_geometries(Reader("./inputDATA/shp/pan_Amazon_mask.shp").geometries(),
                  ccrs.PlateCarree(),
                  facecolor='whitesmoke', hatch='/////////', zorder=1, linewidth=0.2)
    cbar = fig.colorbar(imsh, ax=ax, orientation='vertical',  spacing='proportional', shrink=srk, pad=0)
    cbar.ax.set_ylabel(units)
    cbar.minorticks_on()
    ax.add_feature(mask, edgecolor='k', linewidth=1, facecolor="None", zorder=2.5)
    ax.add_feature(cfeature.BORDERS, edgecolor="grey", zorder=0.5)
    ax.coastlines(resolution='110m', linewidth=1, edgecolor="grey", facecolor="grey", alpha=0.5)

    ax.legend(handles=legend_elements, loc=("best"))
    # if sys.platform == 'linux':
    if vname != "mineral_p":
        ax.annotate("%s%s%s"%(nmodels,"\n", Amean), xy=(_x_, _y_), xycoords='figure points', horizontalalignment='left',
            verticalalignment='bottom', fontsize="small")

    if frac_of_total is not None:
        ax1 = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())
        ax1.set_xticks([-70, -45], crs=ccrs.PlateCarree())
        #ax1.set_yticks([-17, 10], crs=ccrs.PlateCarree())
        ax1.xaxis.set_major_formatter(lon_formatter)
        ax1.set_extent(FOCUS_EXTENT, crs=ccrs.PlateCarree())
        imsh1 = plt.imshow(np.ma.masked_array(frac_of_total, mask=DI_mask),
                           transform=img_proj, extent=img_extent, cmap='cividis', zorder=3)
        ax1.add_geometries(Reader("./inputDATA/shp/pan_Amazon_mask.shp").geometries(),
                  ccrs.PlateCarree(),
                  facecolor='whitesmoke', hatch='/////////', zorder=1, linewidth=0.2)
        cbar1 = fig.colorbar(imsh1, ax=ax1, orientation='vertical',  spacing='proportional', shrink=srk, pad=0)
        cbar1.ax.set_ylabel("%")
        cbar1.minorticks_on()
        ax1.add_feature(mask, edgecolor='k', linewidth=1, facecolor="None", zorder=2.5)
        ax1.add_feature(cfeature.BORDERS, edgecolor="grey", zorder=0.5)
        ax1.coastlines(resolution='110m', linewidth=1, edgecolor="grey", alpha=0.5, zorder=0.5)
        ax1.set_title("Fraction of Total P", fontsize='medium')
        if not vname == 'mineral_p':
            ax2 = fig.add_subplot(gs[0, 2], projection=ccrs.PlateCarree())
            ax2.set_xticks([-70, -45], crs=ccrs.PlateCarree())
            #ax2.set_yticks([-17, 10], crs=ccrs.PlateCarree())
            ax2.xaxis.set_major_formatter(lon_formatter)
            ax2.set_extent(FOCUS_EXTENT, crs=ccrs.PlateCarree())
            ax2.add_geometries(Reader("./inputDATA/shp/pan_Amazon_mask.shp").geometries(),
                  ccrs.PlateCarree(),
                  facecolor='whitesmoke', hatch='/////////', zorder=1, linewidth=0.2)
            imsh1 = plt.imshow(np.ma.masked_array(SE, mask=DI_mask), transform=img_proj, extent=img_extent, cmap='viridis', zorder=3)
            cbar1 = fig.colorbar(imsh1, ax=ax2, orientation='vertical',  spacing='proportional', shrink=srk, pad=0)
            cbar1.ax.set_ylabel(units)
            cbar1.minorticks_on()
            ax2.add_feature(mask, edgecolor='k', linewidth=1, facecolor="None", zorder=2.5)
            ys, xs = pt.T
            # ax2.plot(xs, ys, transform=ccrs.PlateCarree(),
            #         marker=concentric_circle, color='r', markersize=4, linestyle='', zorder=4)
            # ax2.legend(handles=[Line2D([0], [0], linewidth=0.0, color='r',
            #                                     marker=concentric_circle, label='Soil samples',
            #                                     markerfacecolor='r')])
            ax2.add_feature(cfeature.BORDERS, edgecolor="grey", zorder=0.5)
            ax2.coastlines(resolution='110m', linewidth=1, edgecolor="grey", alpha=0.5)
            ax2.set_title("Standard Error", fontsize='medium')

    elif vname == 'total_p':
        ax1 = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())
        #  fig.add_subplot(1, 2, 2, projection=ccrs.PlateCarree())
        ax1.set_xticks([-70, -45], crs=ccrs.PlateCarree())
        #ax1.set_yticks([-17, 10], crs=ccrs.PlateCarree())
        ax1.xaxis.set_major_formatter(lon_formatter)
        ax1.set_extent(FOCUS_EXTENT, crs=ccrs.PlateCarree())
        ax1.add_geometries(Reader("./inputDATA/shp/pan_Amazon_mask.shp").geometries(),
                  ccrs.PlateCarree(),
                  facecolor='whitesmoke', hatch='/////////', zorder=1, linewidth=0.2)
        ys, xs = pt.T
        ax1.plot(xs, ys, transform=ccrs.PlateCarree(),
                    marker=concentric_circle, color='r', markersize=4, linestyle='', zorder=4)
        ax1.legend(handles=[Line2D([0], [0], linewidth=0.0, color='r',
                                                marker=concentric_circle, label='Soil Samples',
                                                markerfacecolor='m')])
        imsh1 = plt.imshow(np.ma.masked_array(SE, mask=DI_mask), transform=img_proj, extent=img_extent, cmap='viridis', zorder=3)
        cbar1 = fig.colorbar(imsh1, ax=ax1, orientation='vertical',  spacing='proportional', shrink=srk, pad=0)
        cbar1.ax.set_ylabel(units)
        cbar1.minorticks_on()
        ax1.add_feature(mask, edgecolor='k', linewidth=1, facecolor="None", zorder=2.5)
        ax1.add_feature(cfeature.BORDERS, edgecolor="grey")
        ax1.coastlines(resolution='110m', linewidth=1, edgecolor="grey", alpha=0.5)
        ax1.set_title("Standard Error", fontsize='medium')

    plt.tight_layout()
    plt.savefig("./p_figs/%s.png" %vname, dpi=750)
    plt.close(fig)


def bplot_perm_imp_avg2():
    """PLOT Permutation importances (AVG -> n == 120 permutations for each model) AKA Mean Decrease in Accuracy"""
    plt.rc('legend', fontsize=6)
    plt.rc('font', size=8) #controls default text size
    plt.rc('axes', titlesize=8) #fontsize of the title
    plt.rc('axes', labelsize=8) #fontsize of the x and y labels
    plt.rc('xtick', labelsize=8) #fontsize of the x tick labels
    plt.rc('ytick', labelsize=8) #fontsize of the y tick labels

    def soil_column(csv_path, old_data):
        data = pd.read_csv(csv_path, usecols=SOIL_FEATURES)
        new_data = []
        x = data.shape[0]
        dt = data.iloc(0)
        for i in range(x):
            new_data.append(dt[i][:].sum())
        old_data['RSG'] = np.array(new_data, dtype=np.float32)
        return old_data

    titles = ['Total P', 'Available P', 'Organic P', 'Inorganic P', "Occluded P"]

    data1 = pd.read_csv("./MDA/permutation_importances_avg_total_p.csv", usecols=NOT_SOIL_FEATURES)
    data1 = soil_column("./MDA/permutation_importances_avg_total_p.csv", data1)

    data2 = pd.read_csv("./MDA/permutation_importances_avg_avail_p.csv", usecols=NOT_SOIL_FEATURES)
    data2 = soil_column("./MDA/permutation_importances_avg_avail_p.csv", data2)

    data3 = pd.read_csv("./MDA/permutation_importances_avg_org_p.csv", usecols=NOT_SOIL_FEATURES)
    data3 = soil_column("./MDA/permutation_importances_avg_org_p.csv", data3)

    data4 = pd.read_csv("./MDA/permutation_importances_avg_inorg_p.csv", usecols=NOT_SOIL_FEATURES)
    data4 = soil_column("./MDA/permutation_importances_avg_inorg_p.csv", data4)

    data5 = pd.read_csv("./MDA/permutation_importances_avg_occ_p.csv", usecols=NOT_SOIL_FEATURES)
    data5 = soil_column("./MDA/permutation_importances_avg_occ_p.csv", data5)

    # return data1, data2
    f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=5, ncols=1, sharex=True, sharey=True)
    f.set_figheight(9.0)
    f.set_figwidth(8.0)

    # AX 1
    ax1.set_ylabel(f"{titles[0]}")
    data1.boxplot(ax=ax1, sym='kx', vert=True, whis=1.5)
    ax1.set_xticklabels(labels='',)
    # AX 2

    ax2.set_ylabel(f"{titles[1]}")
    data2.boxplot(ax=ax2, sym='kx', vert=True, whis=1.5)
    ax2.set_xticklabels(labels='',)
    # AX 3

    ax3.set_ylabel(f"{titles[2]}")
    data3.boxplot(ax=ax3, sym='kx', vert=True, whis=1.5)
    ax3.set_xticklabels(labels='',)
    # AX 4
    ax4.set_ylabel(f"{titles[3]}")
    data4.boxplot(ax=ax4, sym='kx', vert=True, whis=1.5)
    ax4.set_xticklabels(labels='',)

    ax5.set_ylabel(f"{titles[4]}")
    data5.boxplot(ax=ax5, sym='kx', vert=True, whis=1.5)
    # ax4.set_yticklabels(labels='',)
    f.subplots_adjust(top=0.99,
                        bottom=0.03,
                        left=0.08,
                        right=0.98,
                        hspace=0.02,
                        wspace=0.2)

    plt.tight_layout()
    plt.savefig("p_figs/boxplot_perm_importances_avgNS.png", dpi=400)
    plt.close(f)


def bplot_perm_imp_avg3():
    """PLOT Permutation importances (AVG -> n == 120 permutations for each model) AKA Mean Decrease in Accuracy"""
    plt.rc('legend', fontsize=6)
    plt.rc('font', size=8) #controls default text size
    plt.rc('axes', titlesize=8) #fontsize of the title
    plt.rc('axes', labelsize=8) #fontsize of the x and y labels
    plt.rc('xtick', labelsize=8) #fontsize of the x tick labels
    plt.rc('ytick', labelsize=8) #fontsize of the y tick labels


    titles = ['Total P','Available P','Organic P','Inorganic P', "Occluded P"]

    data1 = pd.read_csv("./MDA/permutation_importances_avg_total_p.csv", usecols=SOIL_FEATURES)
    data2 = pd.read_csv("./MDA/permutation_importances_avg_avail_p.csv", usecols=SOIL_FEATURES)
    data3 = pd.read_csv("./MDA/permutation_importances_avg_org_p.csv", usecols=SOIL_FEATURES)
    data4 = pd.read_csv("./MDA/permutation_importances_avg_inorg_p.csv", usecols=SOIL_FEATURES)
    data5 = pd.read_csv("./MDA/permutation_importances_avg_occ_p.csv", usecols=SOIL_FEATURES)

    xlabels = SOIL_FEATURES
    # return data1, data2q
    f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=5, ncols=1, sharex=True, sharey=True)
    f.set_figheight(8.0)
    f.set_figwidth(8.0)

    # AX 1
    ax1.set_ylabel(f"{titles[0]}")
    data1.boxplot(ax=ax1, sym='kx', vert=True, whis=1.5, rot=45)
    ax1.set_xticklabels(labels='',)
    # AX 2
    ax2.set_ylabel(f"{titles[1]}")
    data2.boxplot(ax=ax2, sym='kx', vert=True, whis=1.5, rot=45)
    ax2.set_xticklabels(labels='',)

    # AX 3
    ax3.set_ylabel(f"{titles[2]}")
    data3.boxplot(ax=ax3, sym='kx', vert=True, whis=1.5, rot=45)
    ax3.set_xticklabels(labels='',)

    # AX 4
    ax4.set_ylabel(f"{titles[3]}")
    data4.boxplot(ax=ax4, sym='kx', vert=True, whis=1.5, rot=45)

    ax5.set_ylabel(f"{titles[4]}")
    data5.boxplot(ax=ax5, sym='kx', vert=True, whis=1.5, rot=45)
    ax5.set_xticks(np.arange(1,16))
    ax5.set_xticklabels(labels=xlabels)

    f.subplots_adjust(top=0.99,
                        bottom=0.075,
                        left=0.08,
                        right=0.98,
                        hspace=0.02,
                        wspace=0.2)

    plt.tight_layout()
    plt.savefig("p_figs/boxplot_perm_importances_avgS.png", dpi=400)
    plt.close(f)


def DI_boxplot():
    di_ap =  np.load("./dissimilarity_index_masks/table_DI_AP.npy" )[:, 2]
    di_smp = np.load("./dissimilarity_index_masks/table_DI_SMP.npy")[:, 2]
    di_op =  np.load("./dissimilarity_index_masks/table_DI_OP.npy" )[:, 2]
    di_tp =  np.load("./dissimilarity_index_masks/table_DI_TP.npy" )[:, 2]
    di_ocp =  np.load("./dissimilarity_index_masks/table_DI_OCP.npy" )[:, 2]

    df = np.zeros(shape=(di_ocp.size, 5))

    arrays = [di_tp, di_ap, di_smp, di_op, di_ocp]
    i = 0
    while(i < 5):
        df[:, i] = arrays[i]
        i += 1
    plt.boxplot(df, labels=[ "Total", "Available", "Organic","Inorganic", "Occluded"])
    plt.ylabel("Dissimilarity Index (DI)")
    plt.savefig("./p_figs/DI.png", dpi=300)
    plt.close()
    return df


def plot_eval_metrics(pfrac):
    df = pd.read_csv(f"./model_evaluation_metrics/eval_metrics_{pfrac}.csv")
    df1 = df.drop(["random_state","accuracy", "MAE"], axis=1)
    df2 = df.drop(["random_state","R2", "CV_mean", "CV_std"], axis=1)
    fig, ax = plt.subplots(nrows=1, ncols=2)
    df1.boxplot(ax = ax[1])
    df2.boxplot(ax = ax[0])
    fig.suptitle(f"{labels[pfrac]}")
    plt.savefig(f"./p_figs/eval_metrics_{pfrac}.png", dpi=300)
    plt.close(fig)


def convert(dt, unit_in, unit_out):
    return cfunits.Units.conform(dt, cfunits.Units(unit_in), cfunits.Units(unit_out))


def concat_dfs(aoa="DI_ALL"):
    """concatenate the predictive and observed datasets including a categorical variable to identify the datasets"""
    #select features
    features_obs = ('lat', 'lon', 'Sand', 'Silt', 'Clay', 'Slope', 'Elevation', 'MAT', 'MAP', 'pH', 'TOC', 'TN')
    features_pred = ('lat', 'lon', 'Sand', 'Silt', 'Clay', 'Slope', 'Elevation', 'MAT', 'MAP', 'pH', 'TOC', 'TN')

    obs = pd.read_csv("./inputDATA/fitting_dataset.csv")
    pred = pd.read_csv("./inputDATA/results.csv")

    # FILTER AOA MAP
    # change this to plot with or without AOA-DI
    pred1 = pred.loc[pred[aoa] == 0, :]
    # pred1 = pred
    dt_obs = obs.loc[:, features_obs]
    dt_pred = pred1.loc[:, features_pred]
    # # add srg and dataset columns
    # dt_pred.loc[:, "SRG"] = pred_RSG.SRG
    datasetF = np.array(["OBS" for _ in range(dt_obs.shape[0])])
    datasetP = np.array(["PRED" for _ in range(dt_pred.shape[0])])
    dt_pred.loc[:, "dataset"] = datasetP
    dt_obs.loc[:, "dataset"] = datasetF

    # # correct the index and concat
    n_idx = np.arange(108, 108 + dt_pred.shape[0])
    dt_pred.index = n_idx
    dt = pd.concat([dt_obs, dt_pred], axis=0)
    return dt


def boxplots_SM():
    plt.rc('legend', fontsize=4.8)
    plt.rc('font', size=6) #controls default text size
    plt.rc('axes', titlesize=6) #fontsize of the title
    plt.rc('xtick', labelsize=6) #fontsize of the x tick labels
    plt.rc('ytick', labelsize=6) #fontsize of the y tick labels
    """boxplots comparing the fitting dataset and the predictive dataset"""
    dt = concat_dfs()
    fig, axs = plt.subplots(ncols=4, nrows=3, layout="tight", figsize=(7,4.5))
    variables = ['lat', 'lon', 'Sand', 'Silt', 'Clay', 'Slope', 'Elevation', 'MAT', 'MAP', 'pH', 'TOC', 'TN']
    count = 0
    for row in range(3):
        for col in range(4):
            axin = axs[row, col]
            sns.boxplot(data=dt, x="dataset", y=variables[count], ax=axin)
            axin.set(xlabel="")
            count += 1
    plt.savefig("./p_figs/obs_pred_differences_clean.png", dpi=300)
    plt.close(fig)


def kernplots_DI():
    plt.rc('legend', fontsize=4.8)
    plt.rc('font', size=6) #controls default text size
    plt.rc('axes', titlesize=6) #fontsize of the title
    plt.rc('xtick', labelsize=6) #fontsize of the x tick labels
    plt.rc('ytick', labelsize=6) #fontsize of the y tick labels
    "kde plot comparing hig DI/ low DI"
    dt = pd.read_csv("./inputDATA/results.csv")
    fig, axs = plt.subplots(ncols=4, nrows=3, layout="tight", figsize=(7,4.5))
    variables = ['lat', 'lon', 'Sand', 'Silt', 'Clay', 'Slope', 'Elevation', 'MAT', 'MAP', 'pH', 'TOC', 'TN']
    count = 0
    for row in range(3):
        for col in range(4):
            axin = axs[row, col]
            if variables[count] == "TN":
                sns.kdeplot(data=dt, x=variables[count], hue="DI_ALL", ax=axin, common_norm=False)
            else:
                sns.kdeplot(data=dt, x=variables[count], hue="DI_ALL", ax=axin, legend=None, common_norm=False)
            axin.set(xlabel=variables[count], ylabel="")
            count += 1
    plt.legend(title="DI", labels=["L", "H"])
    plt.savefig("./p_figs/DI_inout_kern_dens.png", dpi=300)
    plt.close(fig)


##PDP & ICE plots
def pdp_plots():
    features = pd.read_csv("./inputDATA/fitting_dataset.csv")
    pforms = ['Inorganic P', 'Organic P',
            'Available P', 'Total P', 'Occluded P']
    pfracs = ["inorg_p", "org_p", "avail_p", "total_p", "occ_p"]

    names = dict(zip(pfracs, pforms))

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

    def get_importances(label):
        return pd.read_csv(f"./MDA/permutation_importances_avg_{label}.csv").mean()

    def plotPD(label):#, features_info, vars):

        model = best_model(label)
        s1 = "best"

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
        features_info = {"features": [x[0] for x in most_important(imp=imp, nloop=6)],
                        "kind": "both"}
        fig, ax = plt.subplots(nrows=3, ncols=2, constrained_layout=True)

        display = PartialDependenceDisplay.from_estimator(
            rf,
            train_features,
            **features_info,
            ax=ax)
        display.figure_.suptitle(f"Partial dependence of {names[label]}", fontsize=12)
        for axes in ax.flat[1:]:
            axes.legend().remove()

        plt.savefig(f"./p_figs/ICE_plot_{label}_{s1}_2_.png", dpi=300)
        plt.close(fig)

    for v in pfracs:
        plotPD(v)

# Stats used in the description

def pair_grid_elev():
    """scatterplots of variables related with elevation"""
    dt = pd.read_csv("./inputDATA/fitting_dataset.csv")
    y_vars = ["Elevation"]
    x_vars = ["Slope", "MAT", "TOC", "TN"]
    g = sns.PairGrid(data=dt, x_vars=x_vars, y_vars=y_vars)
    g.map_offdiag(sns.scatterplot)
    g.add_legend()
    plt.savefig("./p_figs/elev_correlation.png", dpi=300)
    plt.show()

def print_stats_P_dataset():
    """print statistics of the P forms in the P dataset"""
    obs = pd.read_csv("./inputDATA/fitting_dataset.csv")
    stats = obs.describe().loc["mean",:]
    print("----States of the fitting dataset -----")
    print(obs.describe().loc["mean",:])

    print("TP (mg kg-1)", stats.total_p)
    print(f"PROP %: OCC ={round(((stats.occ_p + stats.mineral_p) / stats.total_p) * 100, 2)}")
    print(f"PROP %: OP ={round((stats.org_p / stats.total_p) * 100, 2)}")
    print(f"PROP %: IP ={round((stats.inorg_p/ stats.total_p) * 100, 2)}")
    print(f"PROP %: AP ={round((stats.avail_p / stats.total_p) * 100, 2)}")

def get_area_tot(form="ALL"):
    """get the areas excluded in the Di analysis"""

    area = pa_area
    mask_aoa = get_DI_mask(form)

    total_area = area.sum()
    aoa = np.ma.masked_array(area, mask_aoa).sum()
    print(f"The total area of the pan-Amazon is {convert(total_area, 'm2', 'km2')} km2")
    print(f"The area after the exclusion of the high DI cells for {form} is {convert(aoa, 'm2', 'km2')} km2")
    print(f"Which is {(aoa/total_area * 100)}% of the total area")
    print(f"{(1 - aoa/total_area) * 100} % of the total area was excluded after the DI analysis for {form}")
    return (1 - aoa/total_area) * 100

def plot_di_vars():
    plt.rc('legend', fontsize=4.8)
    plt.rc('font', size=6) #controls default text size
    plt.rc('axes', titlesize=6) #fontsize of the title
    plt.rc('xtick', labelsize=6) #fontsize of the x tick labels
    plt.rc('ytick', labelsize=6) #fontsize of the y tick labels
    df = pd.read_csv("./inputDATA/results.csv")
    variables = ['lat', 'lon', 'Sand', 'Silt', 'Clay', 'Slope',
       'Elevation', 'MAT', 'MAP', 'pH', 'TOC', 'TN', 'DI_ALL']

    dt = df.loc[:, variables]
    fig, axs = plt.subplots(ncols=4, nrows=3, layout="tight", figsize=(7,4.5))

    variables = ['lat', 'lon', 'Sand', 'Silt', 'Clay', 'Slope', 'Elevation', 'MAT', 'MAP', 'pH', 'TOC', 'TN']
    count = 0
    for row in range(3):
        for col in range(4):
            axin = axs[row, col]
            sns.boxplot(data=dt, x="DI_ALL", y=variables[count], ax=axin)
            axin.set(xlabel="", xticklabels=["low", "high DI"])
            count += 1
    plt.savefig("./p_figs/di_boxplot_vars.png", dpi=300)
    plt.close(fig)

def plot_pforms_aoa():
    pforms_names = ['Total P', 'Available P', 'Organic P', 'Inorganic P', 'Occluded P']
    pf_1 = ['total_p', 'avail_p', 'org_p', 'inorg_p', "occ_p"]
    di_1 = ["DI_total_p", "DI_avail_p", "DI_org_p", "DI_inorg_p", "DI_occ_p"]

    """boxplot of p forms for DI/notDI areas"""
    dt = pd.read_csv("./inputDATA/results.csv", index_col="OID_")
    fig, axs = plt.subplots(ncols=5, nrows=1, figsize=(11,4), sharex=True, sharey=True, layout="tight")
    fig.suptitle("")
    for i in range(axs.size):
        ax = axs[i]
        sns.boxplot(data=dt, y=pf_1[i], x=di_1[i], ax=ax)
        if i == 0:
            ax.set(ylabel="mg kg⁻¹")
        else:
            ax.set(ylabel="")
        ax.set(xticklabels=["low", "high DI"], xlabel=pforms_names[i])
    plt.savefig("./p_figs/Pforms_DI.png", dpi=300)

def stats_pforms_table(form):
    """print mean min and max concentrations (mg kg-1) for each P form in the DI cleaned area"""
    df = pd.read_csv("./inputDATA/results.csv")
    # return df
    if form == "mineral_p":
        sel = "ALL"
    else:
        sel = form

    mask = df[f"DI_{sel}"] == 0

    print(form)
    print("Mean\t","min\t","max\t")
    print(f"{round(df[form][mask].mean(), 2)}\t{round(df[form][mask].min(), 2)}\t{round(df[form][mask].max(), 2)}")

def stocks_dens_conc(form="total_p"):
    # # area = convert(np.ma.masked_array(area, mask=mask), "m2", "km2")
    a = pa_area


    if form == "mineral_p":
        with Dataset("./results/mineral_p.nc", "r") as fh:
            conc = fh.variables[form][:]
            conc[conc < 0] = 0
        di_mask = get_DI_mask("ALL")
    else:
        di_mask = get_DI_mask(form)
        with Dataset(f"./results/{form}_AVG.nc", "r") as fh:
            conc = fh.variables[form][:]

    with Dataset(f"./results/{form}_area_density.nc", "r") as fh:
        dens = fh.variables[form][:]
        dens[dens < 0] = 0

    comb_mask = np.ma.mask_or(di_mask, np.ma.getmask(dens))
    dens = np.ma.masked_array(dens, mask=comb_mask)
    conc = np.ma.masked_array(conc, mask=comb_mask)
    a = np.ma.masked_array(a, mask=comb_mask)

    stocks = a * dens
    total = convert(stocks.sum(), "g", "Pg")

    return float(total), conc.mean(), dens.mean(), conc.min(), conc.max()

def plot_maps_stats():
    for v in var:
        print(v, end=" : ")
        print("%.2f Pg P, mean concentration %.2f mg kg-1, mean density %.2f kg m-2 (conc min max = %.2f, %.2f)" %stocks_dens_conc(v))
        # print(stats_pforms_table(v))

def stats_tables(dset="fitting"):
    variables = ['lat', 'lon', 'Sand', 'Silt', 'Clay', 'Slope', 'Elevation', 'MAT', 'MAP', 'pH', 'TOC', 'TN']
    if dset == "fitting":
        dt = pd.read_csv("./inputDATA/fitting_dataset.csv")
    else:
        dt = pd.read_csv("./inputDATA/predictive.csv")



    # print(dt.describe().loc[["mean", "std",], :])
    return dt[variables].describe().T.drop(["count"], axis=1)

def compare():
    import pandas as pd
    from scipy.stats import pearsonr
    df = pd.read_csv("./inputDATA/fitting_dataset_comparison.csv")

    # assuming df is already defined
    corr1, pval1 = pearsonr(df['total_p'], df['He_et_al'])
    corr2, pval2 = pearsonr(df['total_p'], df['total_p_RF'])

    print(f"Pearson correlation between total_p and He_et_al: {corr1:.2f}, p-value: {pval1:.2f}")
    print(f"Pearson correlation between total_p and total_p_RF: {corr2:.2f}, p-value: {pval2:.2f}")

if __name__ == "__main__":
    for v in var:
        plot_Pmap(v)
        if v != "mineral_p":
            plot_eval_metrics(v)
    bplot_perm_imp_avg2()
    bplot_perm_imp_avg3()
    DI_boxplot()
    boxplots_SM()
    pdp_plots()
    kernplots_DI()
    pair_grid_elev()
    print_stats_P_dataset()
    get_area_tot()
    get_area_tot("inorg_p")
    get_area_tot("org_p")
    get_area_tot("avail_p")
    get_area_tot("occ_p")
    get_area_tot("total_p")
    plot_di_vars()
    plot_pforms_aoa()
    plot_maps_stats()
    compare()
    pass

