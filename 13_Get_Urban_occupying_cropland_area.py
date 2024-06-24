import numpy as np
import plotnine
import xarray as xr
import rioxarray as rxr

from collections import defaultdict
from itertools import product
from rasterio.enums import Resampling
from tqdm.auto import tqdm
from joblib import Parallel, delayed

from helper_func.calculate_GAEZ_stats import bincount_with_mask
from helper_func.parameters import UNIQUE_VALUES


# # Uncoarsen GAEZ mask to the same resolution as LUCC mask
# GAEZ_mask = xr.open_dataset('data/GAEZ_v4/GAEZ_mask.tif')
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
GAEZ_mask = xr.open_dataset('data/GAEZ_v4/GAEZ_mask.tif', chunks='auto')
urban_cells = xr.open_dataset('data/results/step_12_urban_potential_arr_reclass.nc', chunks='auto')['data']
lucc_area = xr.open_dataset('data/LUCC/LUCC_Area_km2.nc', chunks='auto')['data']

cropland_cells = xr.open_dataset('data/LUCC/Norm_CLCD_v01_2019.nc', chunks='auto')['data']
cropland_cells = xr.where(cropland_cells == 1, 1, 0)     # Pixels of cropland are 1, all others are 0


urban_occupy_cropland = xr.where(urban_cells & cropland_cells, 1, 0) 
urban_occupy_cropland = (urban_occupy_cropland * lucc_area).astype('float32')
urban_occupy_cropland = urban_occupy_cropland.rio.write_crs(GAEZ_mask.rio.crs)

GAEZ_uncoarsened = xr.open_dataset('data/results/step_13_GAEZ_uncoarsened.nc', chunks='auto')['id']
GAEZ_uncoarsened = GAEZ_uncoarsened.chunk({
    k:v 
    for k,v in dict(urban_occupy_cropland.chunksizes).items() 
    if k in ['band', 'y', 'x']})


# Calculate the area of cropland occupied by urban expansion
def bincount_chunked(arr, ssp, year, msk):
    return (ssp,year), np.bincount(msk.compute().flatten(), weights=arr.compute().flatten(), minlength=GAEZ_mask['band_data'].size)


# Wrap the calculation into delayed objects
tasks = []
for ssp, year in list(product(urban_occupy_cropland['ssp'].data, urban_occupy_cropland['year'].data)):
    for msk, arr in zip(GAEZ_uncoarsened.data.to_delayed().flatten(), 
                        urban_occupy_cropland.sel(ssp=ssp, year=year).data.to_delayed().flatten()): 
        tasks.append(delayed(bincount_chunked)(arr, ssp, year, msk))


# Loop over the delayed objects and calculate the area of cropland occupied by urban expansion
paralle_obj = Parallel(n_jobs=-1, return_as='generator')

outputs = defaultdict(list)
with tqdm(total=len(tasks)) as pbar:
    for (ssp,year), out in paralle_obj(tasks):
        outputs[ssp,year].append(out)
        pbar.update()           


# Combine the results as a xarray
arrs = []
for (ssp,year), val in outputs.items():
    arr_sum = np.array(val).sum(axis=0)
    arr_sum = arr_sum.reshape(GAEZ_mask['band_data'].shape)
    arr_sum = xr.DataArray(arr_sum, coords={k:GAEZ_mask['band_data'].coords[k] for k in ['band', 'y', 'x']})
    arr_sum = arr_sum.expand_dims({'ssp': [ssp], 'year': [year]})
    arr_sum = arr_sum.rio.write_crs(GAEZ_mask.rio.crs)
    
    arrs.append(arr_sum)
    
urban_occupy_cropland = xr.combine_by_coords(arrs)
urban_occupy_cropland.name = 'data'  


# Subtract the cropland loss for 2020 to align with the model baseline
urban_occupy_cropland = urban_occupy_cropland - urban_occupy_cropland.sel(year=2020)

# Apply mask of the research region
urban_occupy_cropland = urban_occupy_cropland * GAEZ_mask['band_data']

# Save the results
encoding = {'data': {'dtype': 'float32', 'zlib': True, 'complevel': 9}}
urban_occupy_cropland.to_netcdf('data/results/step_13_urban_occupy_cropland.nc', encoding=encoding, engine='h5netcdf')



if __name__ == '__main__':
    
    # Read data
    urban_occupy_cropland = xr.open_dataset('data/results/step_13_urban_occupy_cropland.nc', chunks='auto')['data']
    urban_occupy_cropland = urban_occupy_cropland * GAEZ_mask['band_data']
    mask_province = rxr.open_rasterio('data/GAEZ_v4/Province_mask.tif', chunks='auto')
    
    urban_occupy_stats = bincount_with_mask(mask_province.astype('int8'), urban_occupy_cropland)
    urban_occupy_stats = urban_occupy_stats.rename(columns={'bin': 'Province', 'Value': 'Farmland loss (km2)'})
    urban_occupy_stats['Province'] = urban_occupy_stats['Province'].map(dict(enumerate(UNIQUE_VALUES['Province'])))
    
    plotnine.options.figure_size = (10, 6)
    plotnine.options.dpi = 100
    
    g = (
        plotnine.ggplot() +
        plotnine.geom_line(urban_occupy_stats, 
                           plotnine.aes(x='year', y='Farmland loss (km2)', color='ssp')) +
        plotnine.facet_wrap('~Province', scales='free_y') +
        plotnine.theme_bw()
    )
    
    g.save('data/results/fig_step_13_cropland_loss_from_urban_occupation_km2.svg')
    
    
    
    
    
    
    
    
    
    
    
    