import numpy as np
import pandas as pd
import rasterio
import dask.array as da
import h5py

from helper_func.parameters import HDF_BLOCK_SIZE


# Read the urban data
urban_ds = h5py.File('data/LUCC/Urban_1990_2019.hdf5', 'r')
urban_arr = da.from_array(urban_ds['Array'], 
                          chunks=(1, HDF_BLOCK_SIZE, HDF_BLOCK_SIZE))












