
# This folder contains netCDF files with the original experiment's results

NETCDF4(HDF5) format. CRS=EPSG4326 (WGS84)

The complete code, results and input data can be found [here](https://github.com/jpdarela/Reference_phosphorus_maps_pan-Amazon):

The files are named as follows:

_pform_AVG.nc4_ = Average of the maps predicted by the selected Random Forest models.

_pform_SE.nc4_ = Standard Error of the maps predicted by the selected Random Forest models.

_pform_STD.nc4_ = Standard Deviation of the maps predicted by the selected Random Forest models.

_pform_density.nc4_ = pform_AVG.nc4 converted to the units of grams per square meter.

Where _pform_ can be one of: avail_p, org_p, inorg_p, total_p

For the compounding P forms (avail_p, org_p, inorg_p) there is an extra file:

_pform_percent_of_tot.nc4_ = Corresponding fraction of the predicted Total P.

The _Pmin1_Pocc.nc4_ file is the estimated Primary Mineral P + the Occluded P forms.

The predicted_min-occ_over_total_p.nc4 file is the fraction of total P represented by Pmin1_Pocc.nc4.

The predicted maps for each selected model for each target P form are named predicted\__pform_.nc4.

The file final_dataset_pred_DI_Pforms.tab contains results in tabular format.

Files with the initials DI are rasters marking areas with high associated uncertainties.
