import numpy as np
import xarray as xr
import rioxarray as rxr
import dask.dataframe as dd

from itertools import product
from dask.diagnostics import ProgressBar
from rasterio.enums import Resampling
from tqdm.auto import tqdm
from joblib import Parallel, delayed


# Uncoarsen GAEZ mask to the same resolution as LUCC mask
GAEZ_mask = xr.open_dataset('data/GAEZ_v4/GAEZ_mask.tif')
# GAEZ_cell_id = np.arange(GAEZ_mask['band_data'].size).reshape(GAEZ_mask['band_data'].shape).astype('int64')
# GAEZ_mask['id'] = xr.DataArray(GAEZ_cell_id, coords=GAEZ_mask['band_data'].coords)
# GAEZ_mask = GAEZ_mask.chunk({'y': 160, 'x': 149}).drop_vars('band_data')


# LUCC_mask = xr.open_dataset('data/LUCC/LUCC_Province_mask.nc', chunks='auto')['data']
# LUCC_mask = LUCC_mask.rio.write_crs(GAEZ_mask.rio.crs)

# GAEZ_uncoarsened = GAEZ_mask['id'].rio.reproject_match(LUCC_mask, resampling=Resampling.nearest)
# GAEZ_uncoarsened = GAEZ_uncoarsened.chunk(LUCC_mask.chunksizes)

# encoding = {'id': {'dtype': 'int64', 'zlib': True, 'complevel': 9}}
# GAEZ_uncoarsened.to_netcdf('data/results/step_13_GAEZ_uncoarsened.nc', encoding=encoding, engine='h5netcdf')




# Read data
urban_cells = xr.open_dataset('data/results/step_12_urban_potential_arr_reclass.nc', chunks='auto')
GAEZ_uncoarsened = xr.open_dataset('data/results/step_13_GAEZ_uncoarsened.nc',chunks='auto')['id']
lucc_area = xr.open_dataset('data/LUCC/LUCC_Area_km2.nc', chunks='auto')['data']

cropland_cells = xr.open_dataset('data/LUCC/Norm_CLCD_v01_2019.nc', chunks='auto')
cropland_cells = xr.where(cropland_cells == 1, 1, 0)     # Pixels of cropland are 1, all others are 0

urban_occupy_cropland = xr.where(urban_cells & cropland_cells, 1, 0)
urban_occupy_cropland = (urban_occupy_cropland * lucc_area).astype('float32')
urban_occupy_cropland = urban_occupy_cropland.rio.write_crs(GAEZ_mask.rio.crs)



paralle_obj = Parallel(n_jobs=20, prefer='threads', return_as='generator')

def reproject_match(ds, ssp, year, target, resampling=Resampling.sum):
    return (ds,year), ds.sel(ssp=ssp, year=year).rio.reproject_match(target, resampling=resampling)

tasks = []
for ssp, year in list(product(urban_occupy_cropland['ssp'].data, urban_occupy_cropland['year'].data)):
    tasks.append(delayed(reproject_match)(urban_occupy_cropland, ssp, year, GAEZ_mask, Resampling.sum))

outputs = []
with tqdm(total=len(tasks)) as pbar:
    for out in paralle_obj(tasks):
        outputs.append(out)
        pbar.update()  




