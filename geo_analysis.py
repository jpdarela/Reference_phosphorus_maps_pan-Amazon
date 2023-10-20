# geo_analysis.py
from math import sqrt
import numpy as np
from scipy import stats
from netCDF4 import Dataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cfunits
from plot_maps import percentile_Treshold, make_all_mask, make_DI_mask

def find_coord(N, W):
    """ Given a pair of geographic (WGS84) coordinates (decimal degrees)
        returns the Y and X indices in the array (360,720//0.5° lon-lat)
        (C_contiguous) Tested only in south america"""
    Yc = round(N, 2)
    Xc = round(W, 2)

    if abs(Yc) > 89.75:
        if Yc < 0:
            Yc = -89.75
        else:
            Yc = 89.75

    if abs(Xc) > 179.75:
        if Xc < 0:
            Xc = -179.75
        else:
            Xc = 179.75

    Yind = 0
    Xind = 0

    lon = np.arange(-179.75, 180, 0.5)
    lat = np.arange(89.75, -90, -0.5)

    if True:
        while Yc < lat[Yind]:
            Yind += 1
    # else:
    #     Yind += lat.size // 2
    #     while Yc > lat[Yind]:
    #         Yind += 1
    if Xc <= 0:
        while Xc > lon[Xind]:
            Xind += 1
    else:
        Xind += lon.size // 2
        while Xc > lon[Xind]:
            Xind += 1

    return Yind, Xind

pforms_names = ['Total P', 'Available P', 'Organic P', 'Inorganic P']
pf_1 = ['total_p', 'avail_p', 'org_p', 'inorg_p']
di_1 = ["TP", "AP", "OP", "IP"]
pforms = ['lab P', 'org P', 'sec P']
units = "g m⁻²"

# mask = np.load("./mask_raisg-360-720.npy")
obs = pd.read_csv("./inputDATA/fitting_dataset.csv")
pred = pd.read_csv("./inputDATA/predictor_dataset_AOA.csv")
pred_all = pd.read_csv("./inputDATA/predictor_dataset_AOA_ALL.csv")
pred_RSG = pd.read_csv("./inputDATA/soil_type_with_ID.csv")

# Cell areas in m2
with Dataset("./inputDATA/cell_area.nc", 'r') as fh:
    area = np.flipud(fh.variables['cell_area'][:])

# Soil dry bulk density (km dm-3) 0-30cm profile
with Dataset("./inputDATA/soil_bulk_density.nc", 'r') as fh:
    bdens = fh.variables['b_ds_final'][:]

# TOTAL P
with Dataset("./total_p_density.nc4", "r") as fh:
    rf_total_p = np.flipud(fh.variables['total_p'][:])

with Dataset("./total_p_AVG.nc4", "r") as fh:
    rf_total_p_fr = np.flipud(fh.variables['total_p'][:])

# mask_aoa_tp = np.load("./AOA_TP.npy")
# pt = percentile_Treshold(mask_aoa_tp)
# mask_aoa_tp = mask_aoa_tp > pt
mask = rf_total_p.mask

# Concat datasets

def convert(dt, unit_in, unit_out):
    return cfunits.Units.conform(dt, cfunits.Units(unit_in), cfunits.Units(unit_out))

def concat_dfs(obs=obs, pred=pred, aoa="DI_ALL"):
    """concatenate the predictive and observed datasets including a categorical variable to identify the datasets"""
    #select features
    features_obs = ('lat', 'lon', 'Sand', 'Silt', 'Clay', 'Slope', 'Elevation', 'MAT', 'MAP', 'pH', 'TOC', 'TN', "RSG")
    features_pred = ('lat', 'lon', 'Sand', 'Silt', 'Clay', 'Slope', 'Elevation', 'MAT', 'MAP', 'pH', 'TOC', 'TN')
    pred = pd.read_csv("./inputDATA/results.csv")

    # FILTER AOA MAP
    # change this to plot with or without AOA-DI
    pred1 = pred.loc[pred[aoa] == 0, :]
    # pred1 = pred
    dt_obs = obs.loc[:, features_obs]
    dt_pred = pred1.loc[:, features_pred]

    # add srg and dataset columns
    dt_pred.loc[:, "SRG"] = pred_RSG.SRG
    datasetF = np.array(["OBS" for _ in range(dt_obs.shape[0])])
    datasetP = np.array(["PRED" for _ in range(dt_pred.shape[0])])
    dt_pred.loc[:, "dataset"] = datasetP
    dt_obs.loc[:, "dataset"] = datasetF

    # correct the index and concat
    n_idx = np.arange(108, 108 + dt_pred.shape[0])
    dt_pred.index = n_idx
    dt = pd.concat([dt_obs, dt_pred], axis=0)
    return dt

def boxplots_SM():
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
    "kde plot comparing hig DI/ low DI"
    dt = pd.read_csv("./inputDATA/predictor_dataset_AOA_ALL.csv")
    fig, axs = plt.subplots(ncols=4, nrows=3, layout="tight", figsize=(7,4.5))
    variables = ['lat', 'lon', 'Sand', 'Silt', 'Clay', 'Slope', 'Elevation', 'MAT', 'MAP', 'pH', 'TOC', 'TN']
    count = 0
    for row in range(3):
        for col in range(4):
            axin = axs[row, col]
            if variables[count] == "TN":
                sns.kdeplot(data=dt, x=variables[count], hue="ALL1", ax=axin, common_norm=False)
            else:
                sns.kdeplot(data=dt, x=variables[count], hue="ALL1", ax=axin, legend=None, common_norm=False)
            axin.set(xlabel=variables[count], ylabel="")
            count += 1
    plt.legend(title="DI", labels=["L", "H"])
    plt.savefig("./p_figs/DI_inout_kern_dens.png", dpi=300)
    plt.close(fig)

def pair_grid_elev():
    """scatterplots of variables related with elevation"""
    dt = pd.read_csv("./inputDATA/fitting_dataset.csv")
    y_vars = ["Elevation"]
    x_vars = ["Slope", "MAT", "TOC", "TN", "MAP"]
    g = sns.PairGrid(data=dt, x_vars=x_vars, y_vars=y_vars)
    g.map_offdiag(sns.scatterplot)
    g.add_legend()
    plt.savefig("./p_figs/elev_correlation.png", dpi=300)
    plt.show()

def print_stats_P_dataset(obs=obs):
    """print statistics of the P forms in the P dataset"""
    stats = obs.describe().loc["mean",:]
    print(obs.describe().loc["mean",:])
    print("\nTP", stats.total_p)
    print(f"\nPROP: OCC ={round(((stats.occ_p + stats.mineral_p) / stats.total_p) * 100, 2)}")
    print(f"\nPROP: OP ={round((stats.org_p / stats.total_p) * 100, 2)}")
    print(f"\nPROP: IP ={round((stats.inorg_p/ stats.total_p) * 100, 2)}")
    print(f"\nPROP: AP ={round((stats.avail_p / stats.total_p) * 100, 2)}")

def print_corr_df(obs=obs):
    """print spearman rho for some under sampled variables"""
    print(obs.loc[:, ["Elevation","Slope","TOC","TN","MAT"]].corr(method="spearman"))

def get_area_tot(form="TP"):
    """get the areas excluded in the Di analysis"""
    pa_area = np.ma.masked_array(area, mask=mask)
    if form != "ALL":
        mask_aoa = np.load(f"./AOA_{form}.npy")
        pt = percentile_Treshold(mask_aoa)
        mask_aoa = mask_aoa > pt
        ex_area = np.ma.masked_array(pa_area, mask=np.logical_not(mask_aoa))
    else:
        mask_aoa = make_all_mask()
        ex_area = np.ma.masked_array(pa_area, mask=np.logical_not(mask_aoa))
    plt.imshow(ex_area)
    plt.show()
    tot_area = pa_area.sum()
    exc_area = ex_area.sum()
    final_area = cfunits.Units.conform(tot_area, cfunits.Units("m2"), cfunits.Units("km2"))
    final_ex_area = cfunits.Units.conform(exc_area, cfunits.Units("m2"), cfunits.Units("km2"))
    # return tot_area, exc_area
    print(f"Area total in km2: {final_area}\nexcluded fr: {final_ex_area/final_area}")
    print("%.2e"%(final_area))

def make_AOA_ALLasc():
    """build a raster file with the intersection of all DI masks"""
    mask_aoa = np.logical_not(make_all_mask()).data.astype(np.int32)
    idx = np.where(mask == True)
    mask_aoa[idx] = -9999
    header = "ncols 720\nnrows 360\nxllcorner -180\nyllcorner -90\ncellsize 0.5\nnodata_value -9999"
    np.savetxt("AOA_ALL.asc", mask_aoa, fmt="%d", header=header, comments="")
    return mask_aoa

def plot_di_vars():
    df = pd.read_csv("./inputDATA/predictor_dataset_AOA_ALL.csv")
    variables = ['lat', 'lon', 'Sand', 'Silt', 'Clay', 'Slope',
       'Elevation', 'MAT', 'MAP', 'pH', 'TOC', 'TN', 'ALL1']
    dt = df.loc[:, variables]

    fig, axs = plt.subplots(ncols=4, nrows=3, layout="tight", figsize=(7,4.5))

    variables = ['lat', 'lon', 'Sand', 'Silt', 'Clay', 'Slope', 'Elevation', 'MAT', 'MAP', 'pH', 'TOC', 'TN']
    count = 0
    for row in range(3):
        for col in range(4):
            axin = axs[row, col]
            sns.boxplot(data=dt, x="ALL1", y=variables[count], ax=axin)
            axin.set(xlabel="", xticklabels=["E", "NE"])
            count += 1
    plt.savefig("./p_figs/di_boxplot_vars.png", dpi=300)
    plt.close(fig)

def stocks_dens_conc(form="total_p"):
    # area = convert(np.ma.masked_array(area, mask=mask), "m2", "km2")
    a = np.ma.masked_array(area, mask=mask)
    if form == "mineral_p":
        with Dataset("./Pmin1_Pocc.nc4", "r") as fh:
            conc = np.flipud(fh.variables[form][:])
    else:
        with Dataset(f"./{form}_AVG.nc4", "r") as fh:
            conc = np.flipud(fh.variables[form][:])

    with Dataset(f"./{form}_density.nc4", "r") as fh:
        dens = np.flipud(fh.variables[form][:])

    mask1 = make_DI_mask(form)
    multiplier = np.logical_not(mask1).astype(np.int64)

    stocks = a * dens * multiplier
    total = convert(stocks.sum(), "g", "Pg")
    conc_m = np.ma.masked_array(conc, mask1)
    dens_m = np.ma.masked_array(dens, mask1)

    return float(total), conc_m.mean(), dens_m.mean()

def print_stats_pforms():
    """print stocks and mean density & concentration"""
    values = []
    for v in ['total_p', 'org_p', 'inorg_p', 'avail_p', "mineral_p"]:
        values.append(list(map(lambda x: round(x,4), stocks_dens_conc(v))) + [v,])
    return values

def merge_soil_type_to_prfinal():
    """add soil tipes ID and name to the fitting dataset"""
    df1 = pd.read_csv("./inputDATA/predictor_dataset_final.csv")
    df2 = pd.read_csv("./inputDATA/soil_type_with_ID.csv")
    df = pd.merge(df1, df2, on="id")
    df.to_csv("./inputDATA/pred_dataset007.csv")
    return df

def final_df(form="total_p"):
    """include the columns with predicted values of P forms"""
    dt = pd.read_csv("./inputDATA/pred_dataset007.csv")
    soil_code = pd.read_csv("./inputDATA/wrb_soil_type.csv", index_col="ID")
    loop = dt.shape[0]
    fill_ = ["NONE" for _ in range(loop)]
    dt.loc[:,"RSG_2"] = fill_
    data = dt.iloc(0)
    for i in range(loop):
        idx = int(data[i].RSG_ID)
        # print(data[i].RSG,idx, soil_code.loc[idx].NAME)
        dt.loc[i, "RSG_2"] = soil_code.loc[idx].NAME
    dt.to_csv("./inputDATA/final_dataset_pred_DI_Pforms.csv", index=False)
    return dt, soil_code

def plot_pforms_aoa():
    """boxplot of p forms for DI/notDI areas"""
    dt = pd.read_csv("./inputDATA/final_dataset_pred_DI_Pforms.csv", index_col="id")
    fig, axs = plt.subplots(ncols=4, nrows=1, figsize=(11,4), sharex=True, sharey=True, layout="tight")

    for i in range(axs.size):
        ax = axs[i]
        sns.boxplot(data=dt, y=pf_1[i], x=di_1[i], ax=ax)
        if i == 0:
            ax.set(ylabel="mg kg⁻¹")
        else:
            ax.set(ylabel="")
        ax.set(xticklabels=["H", "L"], xlabel=pforms_names[i])
    plt.savefig("./p_figs/Pforms_DI.png", dpi=300)

def stats_pforms_table(form, MASK):
    """print mean min and max concentrations (mg kg-1) for each P form in the DI cleaned area"""
    df = pd.read_csv("./inputDATA/final_dataset_pred_DI_Pforms.csv")
    mask = df[MASK] == 1
    print(form)
    print("Mean\t","min\t","max\t")
    print(f"{round(df[form][mask].mean(), 2)}\t{round(df[form][mask].min(), 2)}\t{round(df[form][mask].max(), 2)}")

def model_perf(form):
    """model perfrmance metrics"""
    ds = pd.read_csv(f"eval_metrics_{form}.csv", index_col="random_state")
    return ds

def print_DIarea_percent():

    """Print areas of regions excluded by """
    df = pd.read_csv("./inputDATA/final_dataset_pred_DI_Pforms.csv")
    masks = []
    for mk in ["ALL", "TP", "AP","OP","IP"]:
        if mk == "ALL":
            total_area = df.area.sum()
            message = f"\nTotal area = {total_area} m²"
        else:
            mask = df[mk] == 1
            pf_area = df.area[mask].sum()
            masks.append(np.logical_not(mask.__array__()))
            print(f"Percentage of area excluded by DI ::{mk}:: {1 - (pf_area / df.area.sum())}\n\t Area: {pf_area} m²\n")
    final_mask = np.logical_or.reduce(masks)
    print(f"{message} of wich:  {(df.area[final_mask].sum() / total_area)} % \nwas excluded due to low dissimilarity")
    return masks

def excluded_RSG_count_area():
    """gridcell counts and relative areas for soil types"""
    dt = pd.read_csv("./inputDATA/final_dataset_pred_DI_Pforms.csv")
    obs = pd.read_csv("./inputDATA/fitting_dataset.csv")

    # Soil types in the fitting dataset
    obs_RSG_counts = obs.SRG.value_counts()

    # Excluded gridcell in the analysis
    mk = dt.ALL == 0

    # Soil types of gridcells that are excluded
    excl_RSG = dt.RSG_2[mk].value_counts()

    # Soil types in the predictive dataset
    pred_dataset_RSG = dt.RSG_2.value_counts()

    # Get undef data
    mk2 = dt.RSG == "Undef"
    undef = dt.RSG_2[mk2].value_counts()



    counts = {"PRED": pred_dataset_RSG,
               "EXCL": excl_RSG,
               "OBS": obs_RSG_counts,
               "UNDEF": undef}

    soil_counts = pd.DataFrame(counts)
    soil_counts.to_csv("descr_stats_soil_counts.csv")
    return soil_counts

def weathering_stage(pt=False):

    w1 = ("Andosols", 'Cambisols',  'Regosols', 'Gleysols', 'Fluvisols', 'Luvisols', 'Umbrisols')
    w2 = ('Alisols', 'Lixisols',  'Nitisols', 'Plinthosols')
    w3 = ( 'Acrisols', 'Arenosols', 'Ferralsols', 'Podzol')

    df = pd.read_csv("./inputDATA/fitting_dataset.csv")
    df2 = df.loc[:,:][df.SRG == "Cambisols"]

    stypes = df.SRG
    weathering_stage = []

    for soil in stypes:
        if soil in w1:
            weathering_stage.append("W1")
        elif soil in w2:
            weathering_stage.append("W2")
        elif soil in w3:
            weathering_stage.append("W3")
        else: assert False
    df.loc[:, "weathering_stage"] = weathering_stage

    if pt:
        x1 = df.TN.loc[df.weathering_stage == "W1"]
        y1 = df.total_p.loc[df.weathering_stage == "W1"]

        x2 = df.TN.loc[df.weathering_stage == "W2"]
        y2 = df.total_p.loc[df.weathering_stage == "W2"]

        x3 = df.TN.loc[df.weathering_stage == "W3"]
        y3 = df.total_p.loc[df.weathering_stage == "W3"]


        res1 = stats.siegelslopes(y1, x1)
        res2 = stats.siegelslopes(y2, x2)
        res3 = stats.siegelslopes(y3, x3)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        sns.scatterplot(data=df, x="TN", y="total_p", hue="weathering_stage", palette=sns.color_palette(["g","b","r"]))
        ax.plot(x1, res1[1] + res1[0] * x1, 'r-')
        ax.plot(x2, res2[1] + res2[0] * x2, 'b-')
        ax.plot(x3, res3[1] + res3[0] * x3, 'g-')
        # ax.plot(df2.TN, df2.total_p, "mx")
        ax.legend(ncol=3)
        kt1 = stats.kendalltau(x1,y1)
        kt2 = stats.kendalltau(x2,y2)
        kt3 = stats.kendalltau(x3,y3)

        ax.text(1.0, 250, f"Kendall τ {round(kt1[0], 2)} p = {round(kt1.pvalue,3)}", color="r") # ; p = {round(kt1[1], 7)}
        ax.text(1.0, 150, f"Kendall τ {round(kt2[0], 2)} p = {round(kt2.pvalue,3)}", color="b") # ; p = {round(kt2[1], 7)}
        ax.text(1.0,  50, f"Kendall τ {round(kt3[0], 2)} p = {round(kt3.pvalue,3)}", color="g") # ; p = {round(kt3[1], 7)}

        ax.set(ylabel="Total P (mg kg⁻¹)", xlabel="Total N (%)")
        plt.show()
    # plt.savefig("./p_figs/TEST2.png", dpi=600)
    # plt.close(fig)
    return kt1

def kt_tp_tn():
    df = pd.read_csv("./inputDATA/fitting_dataset.csv")
    x = df.TN
    y = df.total_p
    res = stats.siegelslopes(y, x)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    sns.scatterplot(data=df, x="TN", y="total_p", hue="SRG", palette=sns.color_palette("husl", 15))
    ax.plot(x, res[1] + res[0] * x, 'r-')
    ax.legend(bbox_to_anchor=(0.55, 0.51), ncol=2)
    kt = stats.kendalltau(x,y)
    ax.text(1.0, 200, f"Kendall τ {round(kt[0], 2)}; p < 0.05")
    ax.set(ylabel="Total P (mg kg⁻¹)", xlabel="Total N (%)")
    plt.savefig("./p_figs/TEST.png", dpi=600)
    plt.close(fig)

    return df

def tp_tn_weathering_stages():
    df = weathering_stage()
    fig, ax = plt.subplots(ncols=1, nrows=3, layout="tight")
    sns.barplot(data=df, y="weathering_stage", x="TN", ax=ax[1])
    sns.barplot(data=df, y="weathering_stage", x="total_p", ax=ax[0])
    sns.barplot(data=df, y="weathering_stage", x="TOC", ax=ax[2])
    plt.show()

def std_table():
    df = weathering_stage()

def compare_with_He2021():
    dt = Dataset("../total_P_0-30cm_he_etal2021.nc")
    he = dt.variables["Band1"][...]
    dt.close()
    dt = Dataset("./total_p_AVG.nc4")
    rf = dt.variables["total_p"][...]
    dt.close()
    return rf, he


def comp_3():
    dt = pd.read_csv("./results_comp.csv")
    msk = dt.TP == 1
    rf = dt.total_p.loc[msk]
    he = dt.tp_he.loc[msk]
    ya = dt.tp_yang[msk]

def add_obs():
    for x in range(obs.shape[0]):
        obs_data = obs.iloc(0)[x]
        print(obs_data.lat)

def plot_comp_corrs():
    dt = pd.read_csv("./fitting_comparison.csv")
    print(dt.loc[:, ["total_p","yang2013","he2021","darela2023"]].corr(method="pearson"))

def plot_comp_he_2021(f=stats.pearsonr):

    dt = pd.read_csv("./fitting_comparison.csv")

    x  = dt.total_p
    y1 = dt.darela2023
    y2 = dt.he2021

    print("da", f(y1,x))
    print("re", f(y2,x))
    return dt