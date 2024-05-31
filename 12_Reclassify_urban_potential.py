import dask.array as da
import pandas as pd
import h5py
import rioxarray as rxr
import rioxarray
import xarray as xr

from affine import Affine
from helper_func.parameters import BLOCK_SIZE, UNIQUE_VALUES

work_size = BLOCK_SIZE * 8

# Read data
potential_threshold = pd.read_csv('data/results/step_11_potential_threshold.csv').drop(columns=['Area_cumsum_km2'])
potential_threshold = xr.DataArray.from_series(potential_threshold.set_index(['Province', 'ssp', 'year']).squeeze())

region_arr = xr.open_dataset('data/LUCC/LUCC_Province_mask.nc', chunks={'y': work_size, 'x': work_size})['data']
region_arr = xr.where(region_arr<0, 0, region_arr).astype(int)
region_arr = [xr.where(region_arr == idx, 1, 0).expand_dims({'Province': [i]}) for idx,i in enumerate(UNIQUE_VALUES['Province'])]
region_arr = xr.combine_by_coords(region_arr)['data'].astype('float32')

urban_potential_arr = xr.open_dataset('data/LUCC/Norm_Transition_potential.nc', chunks={'y': work_size, 'x': work_size})['data'].astype('float32')
urban_potential_arr = region_arr * urban_potential_arr 



# Reclassify urban potential
urban_potential_arr_reclass = xr.where(urban_potential_arr < potential_threshold, 0, 1).astype('int8')
urban_potential_arr_reclass = urban_potential_arr_reclass.sum(dim='Province')


urban_potential_arr_reclass[0,...,0,10][::100,::100].plot()












