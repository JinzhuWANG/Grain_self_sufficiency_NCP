import pandas as pd
import xarray as xr

from helper_func.parameters import BLOCK_SIZE, UNIQUE_VALUES

work_size = BLOCK_SIZE * 8

# Read data
potential_threshold = pd.read_csv('data/results/step_11_potential_threshold.csv').drop(columns=['Area_cumsum_km2'])
potential_threshold = xr.DataArray.from_series(potential_threshold.set_index(['Province', 'SSP', 'year']).squeeze())

region_arr = xr.open_dataset('data/LUCC/LUCC_Province_mask.nc', chunks={'y': work_size, 'x': work_size})['data']
region_arr = xr.where(region_arr<0, 0, region_arr).astype(int)
region_arr = [xr.where(region_arr == idx, 1, 0).expand_dims({'Province': [i]}) for idx,i in enumerate(UNIQUE_VALUES['Province'])]
region_arr = xr.combine_by_coords(region_arr)['data'].astype('float32')

urban_potential_arr = xr.open_dataset('data/LUCC/Norm_Transition_potential.nc', chunks={'y': work_size, 'x': work_size})['data'].astype('float32')
urban_potential_arr = region_arr * urban_potential_arr 



# Reclassify urban potential
urban_potential_arr_reclass = xr.where(urban_potential_arr < potential_threshold, 0, 1)
urban_potential_arr_reclass = urban_potential_arr_reclass.sum(dim='Province').astype('bool')
urban_potential_arr_reclass = urban_potential_arr_reclass.transpose('SSP', 'year','band','y', 'x')

urban_potential_arr_reclass.name = 'data'
encoding = {'data': {'dtype': 'bool', 'zlib': True, 'complevel': 9}}
urban_potential_arr_reclass.to_netcdf('data/results/step_12_urban_potential_arr_reclass.nc', encoding=encoding, engine='h5netcdf')







