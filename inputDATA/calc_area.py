from netCDF4 import Dataset
import numpy as np
from ILAMB.ilamblib import CellAreas


with Dataset("extent.nc") as dt:
    lon = dt.variables["lon"][:]
    lat = dt.variables["lat"][:]
    mask = dt.variables["Band1"][:].mask

area = CellAreas(lat, lon)

np.savez("extent.npz", area=area, lat=lat, lon=lon, cell_size=1/12)