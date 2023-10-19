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
import sys
import os

import numpy as np
import pandas as pd
import cfunits
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
from netCDF4 import Dataset

from table2raster import save_nc, ydim, xdim, cell_size, lat, lon

from scientific_colourmaps import gm2_map as gm2
from scientific_colourmaps import batlowK_map as batlowK

FILE = 7

EXTENT = [lon[ 0] - cell_size / 2,
          lon[-1] + cell_size / 2,
          lat[-1] - cell_size / 2,
          lat[ 0] + cell_size / 2]

os.makedirs("p_figs", exist_ok=True)

legend_elements = [Line2D([0], [0], color='k', lw=1, label='Pan-Amazon'),]

pforms = ['Inorganic P', 'Organic P', 'Available P',
          'Total P', 'Occluded P', 'Primary mineral P']

var = ['inorg_p', 'org_p', 'avail_p', 'total_p', 'occ_p', 'mineral_p']

labels = dict(list(zip(var, pforms)))

NOT_SOIL_FEATURES = ["lat", "lon", "Sand", "Silt", "Clay", "Slope",
                     "Elevation", "MAT", "MAP", "pH", "TOC", "TN"]

SOIL_FEATURES = ["SRG_Acrisols", "SRG_Alisols", "SRG_Andosols", "SRG_Arenosols",
                 "SRG_Cambisols", "SRG_Ferralsols", "SRG_Fluvisols", "SRG_Gleysols",
                 "SRG_Lixisols", "SRG_Luvisols", "SRG_Nitisols", "SRG_Plinthosols",
                 "SRG_Podzol","SRG_Regosols","SRG_Umbrisols"]

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
    plt.rc('legend', fontsize=6)
    plt.rc('font', size=8) #controls default text size
    plt.rc('axes', titlesize=8) #fontsize of the title
    plt.rc('axes', labelsize=7) #fontsize of the x and y labels
    plt.rc('xtick', labelsize=8) #fontsize of the x tick labels
    plt.rc('ytick', labelsize=7) #fontsize of the y tick labels

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
        srk  = 0.77
        _x_= 49.5
        _y_= 40
    else:
        fig = plt.figure(figsize=(8.2, 2.6))
        gs = gridspec.GridSpec(1, 3)
        srk = 0.754 # 0.763 # For org_p # AVAIL and INORG = 0.0.765
        _x_= 75
        _y_= 40

    # Main Map
    ax = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
    ax.set_xticks([-70, -45], crs=ccrs.PlateCarree())
    ax.set_yticks([-16, 10], crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=True, number_format='g')
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)

    ax.set_title(title, fontsize='medium')
    ax.set_extent([-81.5, -41.5, -23.5, 14.5], crs=ccrs.PlateCarree())

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
        ax1.set_extent([-81.5, -41.5, -23.5, 14.5], crs=ccrs.PlateCarree())
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
            ax2.set_extent([-81.5, -41.5, -23.5, 14.5], crs=ccrs.PlateCarree())
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
        ax1.set_extent([-81.5, -41.5, -23.5, 14.5], crs=ccrs.PlateCarree())
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

    if sys.platform == 'linux':
        plt.tight_layout()
        plt.plot()
    plt.savefig("./p_figs/%s.png" %vname, dpi=750)
    plt.close(fig)


# def bplot_perm_imp_avg2():
#     """PLOT Permutation importances (AVG -> n == 120 permutations for each model) AKA Mean Decrease in Accuracy"""
#     plt.rc('legend', fontsize=6)
#     plt.rc('font', size=8) #controls default text size
#     plt.rc('axes', titlesize=8) #fontsize of the title
#     plt.rc('axes', labelsize=8) #fontsize of the x and y labels
#     plt.rc('xtick', labelsize=8) #fontsize of the x tick labels
#     plt.rc('ytick', labelsize=8) #fontsize of the y tick labels

#     def soil_column(csv_path, old_data):
#         data = pd.read_csv(csv_path, usecols=SOIL_FEATURES)
#         new_data = []
#         x = data.shape[0]
#         dt = data.iloc(0)
#         for i in range(x):
#             new_data.append(dt[i][:].sum())
#         old_data['SRG (Sum)'] = np.array(new_data, dtype=np.float32)
#         return old_data

#     titles = ['Total P', 'Available P', 'Organic P', 'Inorganic P']

#     data1 = pd.read_csv("permutation_importances_avg_total_p.csv", usecols=NOT_SOIL_FEATURES)
#     # data1 = soil_column("permutation_importances_avg_total_p.csv", data1)
#     data2 = pd.read_csv("permutation_importances_avg_avail_p.csv", usecols=NOT_SOIL_FEATURES)
#     # data2 = soil_column("permutation_importances_avg_avail_p.csv", data2)
#     data3 = pd.read_csv("permutation_importances_avg_org_p.csv", usecols=NOT_SOIL_FEATURES)
#     # data3 = soil_column("permutation_importances_avg_org_p.csv", data3)
#     data4 = pd.read_csv("permutation_importances_avg_inorg_p.csv", usecols=NOT_SOIL_FEATURES)
#     # data4 = soil_column("permutation_importances_avg_inorg_p.csv", data4)

#     # return data1, data2
#     f, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1, sharex=True, sharey=True)
#     f.set_figheight(8.0)
#     f.set_figwidth(8.0)


#     # AX 1
#     # ax1.set_title(titles[0])
#     #ax1.set_xlabel("MDA")
#     ax1.set_ylabel(f"MDA ({titles[0]})")
#     data1.boxplot(ax=ax1, sym='kx', vert=True, whis=1.5)
#     ax1.set_xticklabels(labels='',)
#     # AX 2
#     # ax2.set_title(titles[1])
#     # ax2.set_xlabel("MDA")
#     ax2.set_ylabel(f"MDA ({titles[1]})")
#     data2.boxplot(ax=ax2, sym='kx', vert=True, whis=1.5)
#     # ax2.set_yticklabels(labels='',)
#     ax2.set_xticklabels(labels='',)
#     # AX 3
#     # ax3.set_title(titles[2])
#     # ax3.set_xlabel("MDA")
#     ax3.set_ylabel(f"MDA ({titles[2]})")
#     data3.boxplot(ax=ax3, sym='kx', vert=True, whis=1.5)
#     # ax3.set_yticklabels(labels='',)
#     ax3.set_xticklabels(labels='',)
#     # AX 4
#     # ax4.set_title(titles[3])
#     # ax4.set_xlabel("Features")
#     ax4.set_ylabel(f"MDA ({titles[3]})")
#     data4.boxplot(ax=ax4, sym='kx', vert=True, whis=1.5)
#     # ax4.set_yticklabels(labels='',)
#     f.subplots_adjust(top=0.99,
#                         bottom=0.03,
#                         left=0.08,
#                         right=0.98,
#                         hspace=0.02,
#                         wspace=0.2)

#     plt.tight_layout()
#     # plt.savefig("p_figs/boxplot_perm_importances_avgNS.tif", dpi=750, pil_kwargs={"compression":"jpeg", "quality":100})
#     plt.savefig("p_figs/boxplot_perm_importances_avgNS.png", dpi=400)
#     plt.close(f)


# def bplot_perm_imp_avg3():
#     """PLOT Permutation importances (AVG -> n == 120 permutations for each model) AKA Mean Decrease in Accuracy"""
#     plt.rc('legend', fontsize=6)
#     plt.rc('font', size=8) #controls default text size
#     plt.rc('axes', titlesize=8) #fontsize of the title
#     plt.rc('axes', labelsize=8) #fontsize of the x and y labels
#     plt.rc('xtick', labelsize=8) #fontsize of the x tick labels
#     plt.rc('ytick', labelsize=8) #fontsize of the y tick labels


#     titles = ['Total P','Available P','Organic P','Inorganic P']

#     data1 = pd.read_csv("permutation_importances_avg_total_p.csv", usecols=SOIL_FEATURES)
#     data2 = pd.read_csv("permutation_importances_avg_avail_p.csv", usecols=SOIL_FEATURES)
#     data3 = pd.read_csv("permutation_importances_avg_org_p.csv", usecols=SOIL_FEATURES)
#     data4 = pd.read_csv("permutation_importances_avg_inorg_p.csv", usecols=SOIL_FEATURES)

#     xlabels = [string.split('_')[-1] for string in SOIL_FEATURES]
#     # return data1, data2q
#     f, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1, sharex=True, sharey=True)
#     f.set_figheight(8.0)
#     f.set_figwidth(8.0)


#     # AX 1
#     # ax1.set_title(titles[0])
#     #ax1.set_xlabel("MDA")
#     ax1.set_ylabel(f"MDA ({titles[0]})")
#     data1.boxplot(ax=ax1, sym='kx', vert=True, whis=1.5, rot=45)
#     ax1.set_xticklabels(labels='',)
#     # AX 2
#     # ax2.set_title(titles[1])
#     # ax2.set_xlabel("MDA")
#     ax2.set_ylabel(f"MDA ({titles[1]})")
#     data2.boxplot(ax=ax2, sym='kx', vert=True, whis=1.5, rot=45)
#     # ax2.set_yticklabels(labels='',)
#     ax2.set_xticklabels(labels='',)
#     # AX 3
#     # ax3.set_title(titles[2])
#     # ax3.set_xlabel("MDA")
#     ax3.set_ylabel(f"MDA ({titles[2]})")
#     data3.boxplot(ax=ax3, sym='kx', vert=True, whis=1.5, rot=45)
#     # ax3.set_yticklabels(labels='',)
#     ax3.set_xticklabels(labels='',)
#     # AX 4
#     # ax4.set_title(titles[3])
#     # ax4.set_xlabel("Features")
#     ax4.set_ylabel(f"MDA ({titles[3]})")
#     data4.boxplot(ax=ax4, sym='kx', vert=True, whis=1.5, rot=45)
#     ax4.set_xticks(np.arange(1,16))
#     ax4.set_xticklabels(labels=xlabels)
#     f.subplots_adjust(top=0.99,
#                         bottom=0.075,
#                         left=0.08,
#                         right=0.98,
#                         hspace=0.02,
#                         wspace=0.2)

#     plt.tight_layout()
#     # plt.savefig("p_figs/boxplot_perm_importances_avgS.tif", dpi=750, pil_kwargs={"compression":"jpeg", "quality":100})
#     plt.savefig("p_figs/boxplot_perm_importances_avgS.png", dpi=400)
#     plt.close(f)


# def plot_points():
#     """PLOT points of the fitting dataset"""
#     plt.rc('legend', fontsize=8)
#     plt.rc('font', size=10) #controls default text size
#     plt.rc('axes', titlesize=11) #fontsize of the title
#     plt.rc('axes', labelsize=10) #fontsize of the x and y labels
#     plt.rc('xtick', labelsize=10) #fontsize of the x tick labels
#     plt.rc('ytick', labelsize=10) #fontsize of the y tick labels


#     theta = np.linspace(0, 2 * np.pi, 100)
#     circle_verts = np.vstack([np.sin(theta), np.cos(theta)]).T
#     concentric_circle = Path.make_compound_path(Path(circle_verts[::-1]),
#                                                 Path(circle_verts * 1))

#     points = []
#     with open('./inputDATA/fitting_dataset.csv', 'r') as fh:
#         reader = csv.reader(fh)
#         for line in reader:
#             points.append(line[2:4])
#     legend_elements = [Line2D([0], [0], color='k', lw=1, label='Pan-Amazon'),]
#     pt = np.array(points[1:][:], dtype=np.float32)

#     fig = plt.figure(figsize=(8, 8))
#     gs = gridspec.GridSpec(1, 1)
#     ax = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())

#     # ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
#     ax.set_xticks([-70, -45], crs=ccrs.PlateCarree())
#     ax.set_yticks([-17, 10], crs=ccrs.PlateCarree())
#     ax.tick_params(labelsize='small')
#     lon_formatter = LongitudeFormatter(zero_direction_label=True, number_format='g')
#     lat_formatter = LatitudeFormatter()
#     ax.xaxis.set_major_formatter(lon_formatter)
#     ax.yaxis.set_major_formatter(lat_formatter)

#     ax.set_extent([-81.5, -41.5, -23.5, 14.5], crs=ccrs.PlateCarree())


#     ax.add_feature(mask, edgecolor='k', linewidth=1, facecolor="None")
#     ys, xs = pt.T
#     ax.plot(xs, ys, transform=ccrs.PlateCarree(),
#             marker=concentric_circle, color='m', markersize=4, linestyle='')
#     ax.add_feature(cfeature.BORDERS, edgecolor="grey")
#     ax.coastlines(resolution='110m', linewidth=1, edgecolor="grey", alpha=0.5)

#     ax.legend(handles=legend_elements + [Line2D([0], [0], linewidth=0.0, color='m',
#                                                 marker=concentric_circle, label='Soil Samples',
#                                                 markerfacecolor='m')])
#     plt.savefig("p_figs/obs_points.pdf")
#     plt.close(fig)


# def ppools_gm2(vname):
#     """Build Netcdfs with values in g.m⁻²(0-300mm). Use the T_BULK_DEN.nc from HWSDv-1.2"""
#     var = ['total_p', 'org_p', 'inorg_p', 'avail_p']
#     if vname in var:
#         dt = Dataset(f"./predicted_{vname}.nc4").variables[vname][:,:,:]
#         form_p = np.flipud(dt.data.mean(axis=0,))
#         mask = form_p == -9999.0
#     else:
#         dt = Dataset("./Pmin1_Pocc.nc4").variables[vname][:,:]
#         form_p = np.flipud(dt.data)
#         mask = form_p == -9999.0

#     bd = Dataset("./inputDATA/soil_bulk_density.nc").variables['b_ds_final'][:]

#     tp = cfunits.Units.conform(form_p, cfunits.Units('mg kg-1'), cfunits.Units('g g-1'))
#     den = cfunits.Units.conform(bd, cfunits.Units('kg dm-3'), cfunits.Units('g m-3'))

#     p_form = np.ma.masked_array((0.3 * tp) * den, mask=mask == True)
#     save_nc(f"{vname}_density.nc4", p_form, vname, ndim=None, units='g.m-2')


# def plot_gm2():
#     """Plot Pfroms in g.m⁻²"""
#     plt.rc('legend', fontsize=6)
#     plt.rc('font', size=10) #controls default text size
#     plt.rc('axes', titlesize=10) #fontsize of the title
#     plt.rc('axes', labelsize=6) #fontsize of the x and y labels
#     plt.rc('xtick', labelsize=6) #fontsize of the x tick labels
#     plt.rc('ytick', labelsize=6) #fontsize of the y tick labels

#     units = 'g m⁻²'

#     var = ['total_p', 'org_p', 'inorg_p', 'avail_p']
#     lab = ['Total P', 'Organic P', 'Inorganic P', 'Available P']
#     # READ_DATA:

#     maps = []
#     for vname, label in zip(var,lab):
#         with Dataset(f"{vname}_density.nc4") as fh:
#             maps.append(np.flipud(fh.variables[vname][...]))

#     # ## ---------PLOT MAP-----------------
#     # Raster extent
#     vminimo = min([np.min(x) for x in maps])
#     vmaximo = max([np.max(x) for x in maps])

#     img_proj = ccrs.PlateCarree()
#     img_extent = [-180, 180, -90, 90]

#     fig = plt.figure(figsize=(8.2, 2))
#     gs = gridspec.GridSpec(1, 4)

#     counter = 0
#     for vname, label in zip(var,lab):
#         msk = make_DI_mask(vname)
#         ax = fig.add_subplot(gs[0, counter], projection=ccrs.PlateCarree())

#         ax.set_xticks([-70, -45], crs=ccrs.PlateCarree())
#         if counter == 0:
#             ax.set_yticks([-17, 10], crs=ccrs.PlateCarree())
#         lon_formatter = LongitudeFormatter(zero_direction_label=True, number_format='g')
#         lat_formatter = LatitudeFormatter()
#         ax.xaxis.set_major_formatter(lon_formatter)
#         if counter == 0:
#             ax.yaxis.set_major_formatter(lat_formatter)

#         ax.set_title(label)
#         ax.set_extent([-81.5, -41.5, -23.5, 14.5], crs=ccrs.PlateCarree())

#         imsh = plt.imshow(np.ma.masked_array(maps[counter], mask=msk), transform=img_proj, extent=img_extent, cmap=gm2, vmin=vminimo, vmax=vmaximo, zorder=6)
#         cbar = fig.colorbar(imsh, ax=ax, orientation='vertical',  spacing='proportional', shrink=0.78, pad=0)
#         if counter == 3:
#             cbar.ax.set_ylabel(units)
#         cbar.minorticks_on()
#         ax.add_feature(mask, edgecolor='k', linewidth=1, facecolor="None", zorder=2.5)
#         ax.add_feature(cfeature.BORDERS, edgecolor="grey", zorder=1.2)
#         ax.coastlines(resolution='110m', linewidth=1, edgecolor="grey", alpha=0.5, zorder=1.5)
#         ax.add_geometries(Reader("./inputDATA/shp/pan_Amazon_mask.shp").geometries(),
#                   ccrs.PlateCarree(),
#                   facecolor='whitesmoke', hatch='/////////', zorder=5, linewidth=0.2)
#         counter += 1
#     # return maps
#     plt.tight_layout()
#     plt.savefig("p_figs/p_pools_gm2.png", dpi=850)
#     plt.close(fig=fig)


# def DI_boxplot():
#     di_ap = pd.read_csv("./DI_AP.csv").T.iloc[2]
#     di_smp = pd.read_csv("./DI_SMP.csv").T.iloc[2]
#     di_op = pd.read_csv("./DI_OP.csv").T.iloc[2]
#     di_tp = pd.read_csv("./DI_TP.csv").T.iloc[2]
#     df = pd.concat([di_ap, di_smp, di_op, di_tp], axis=1)
#     plt.boxplot(df, labels=["Available P", "Inorganic P", "Organic P", "Total P"])
#     plt.ylabel("Dissimilarity Index (DI)")
#     plt.savefig("./p_figs/DI.png", dpi=300)
#     plt.close()


# def plot_eval_metrics(pfrac):
#     df = pd.read_csv(f"eval_metrics_{pfrac}.csv")
#     df1 = df.drop(["random_state","accuracy", "MAE"], axis=1)
#     df2 = df.drop(["random_state","R2", "CV_mean", "CV_std"], axis=1)
#     fig, ax = plt.subplots(nrows=1, ncols=2)
#     df1.boxplot(ax = ax[1])
#     df2.boxplot(ax = ax[0])
#     fig.suptitle(f"{labels[pfrac]}")
#     plt.savefig(f"./p_figs/eval_metrics_{pfrac}.png", dpi=300)
#     plt.close(fig)


if __name__ == "__main__":
    pass

    os.makedirs("./p_figs", exist_ok=True)
    for v in var:
        plot_Pmap(v)
    #         plot_eval_metrics(v)
    #     ppools_gm2(v)
    # bplot_perm_imp_avg2()
    # bplot_perm_imp_avg3()
    # plot_points()
    # plot_gm2()
    # DI_boxplot()
