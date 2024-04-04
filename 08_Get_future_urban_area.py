import numpy as np
import pandas as pd
import rasterio
import dask.array as da
import h5py

from helper_func.parameters import HDF_BLOCK_SIZE


# Read the lucc mask
with h5py.File('data/LUCC/LUCC_Province_mask.hdf5', 'r') as f:
    arr = f['Array']
    lucc_mask_da = da.from_array(arr, chunks=arr.chunks)
    lucc_mask_da_sparsed = lucc_mask_da[:, ::10, ::10].compute()


# Read the urban data
urban_ds = h5py.File('data/LUCC/Urban_1990_2019.hdf5', 'r')
urban_arr = da.from_array(urban_ds['Array'], chunks=urban_ds['Array'].chunks)












