import xarray as xr
import rioxarray as rxr
import pandas as pd

from helper_func.calculate_GAEZ_stats import bincount_with_mask
from helper_func.parameters import UNIQUE_VALUES

# Get masks
mask = rxr.open_rasterio('data/GAEZ_v4/GAEZ_mask.tif')
mask_sum = rxr.open_rasterio('data/GAEZ_v4/province_mask.tif')
mask_province = [(mask_sum == idx).expand_dims({'Province': [p]}) * mask
                    for idx, p in enumerate(UNIQUE_VALUES['Province'])]
mask_province = xr.combine_by_coords(mask_province)


# Read data
yield_MC_yrbook = xr.open_dataset('data/results/step_7_Yearbook_MC_ratio.nc', chunks='auto')['data']
yield_MC_actual = xr.open_dataset('data/results/step_7_GAEZ_yield_MC.nc', chunks='auto')['data']        # t/ha
yield_pred = yield_MC_actual * yield_MC_yrbook


area_start = xr.open_dataset('data/results/step_15_GAEZ_area_km2_adjusted.nc', chunks='auto')['data']   # km2
area_start = area_start * mask_province
area_crop_water_ratio = area_start.sum(dim=['y', 'x', 'band']) / area_start.sum(dim=['crop', 'water_supply', 'y', 'x', 'band'])

area_urban_reduce = xr.open_dataset('data/results/step_13_urban_occupy_cropland.nc', chunks='auto')['data']
area_urban_reduce = area_urban_reduce * area_crop_water_ratio
area_reclaim_increase = xr.open_dataset('data/results/step_14_reclimation_area_km2.nc', chunks='auto')['data']
area_reclaim_increase = area_reclaim_increase * area_crop_water_ratio

area_pred = area_start + area_reclaim_increase - area_urban_reduce
area_pred = area_pred.sum(dim=['Province'])


# Get production
production_pred = yield_pred * area_pred /1e4                           # million tonnes
production_pred_mean = production_pred.mean(dim=['sample'])
production_pred_std = production_pred.std(dim=['sample'])

production_mean_stats = bincount_with_mask(mask_sum, production_pred_mean)
production_std_stats = bincount_with_mask(mask_sum, production_pred_std)
production_stats = pd.merge(
    production_mean_stats, 
    production_std_stats, 
    on = [ 'rcp', 'ssp', 'year', 'bin', 'crop', 'water_supply', 'c02_fertilization'], 
    suffixes = ('_mean', '_std'))
production_stats = production_stats.rename(columns = {'Value_mean': 'Production (tonnes)', 'Value_std': 'Production_std'})


