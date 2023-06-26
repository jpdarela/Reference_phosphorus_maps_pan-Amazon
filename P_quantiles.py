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
import glob
from pathlib import Path
import numpy as np
from netCDF4 import Dataset

FILE = 8

"""Calculate and print the quartiles of the P maps in g(P form) m⁻²"""

files = glob.glob1(os.getcwd(), "*_density.nc4")

for f in files:
    var = "_".join(f.split('_')[0:2])
    fpath = Path(os.path.join(os.getcwd(), f)).resolve()
    print(var, fpath)
    # open the dataset and retrieve raster data as an array
    dataset = Dataset(str(fpath), mode='r').variables[var][:]
    array = dataset.__array__()

    a = array.flatten()
    b = a[a != -9999.0]

    # use the numpy percentile function to calculate percentile thresholds
    Q1 = np.percentile(b, 25)
    Q2 = np.percentile(b, 50)
    Q3 = np.percentile(b, 75)

    print(f, ": Q1, Q2, Q3 -> ", Q1, Q2, Q3)