import math
import numpy as np
import xarray as xr
import rioxarray as rxr
import pandas as pd
import plotnine

from helper_func.calculate_GAEZ_stats import bincount_with_mask, get_GAEZ_stats
from helper_func.get_yearbook_records import get_yearbook_area, get_yearbook_production
from helper_func.parameters import UNIQUE_VALUES, Monte_Carlo_num, BASE_YR


# Get masks
mask = rxr.open_rasterio('data/GAEZ_v4/GAEZ_mask.tif')
mask_mean = rxr.open_rasterio('data/GAEZ_v4/Province_mask_mean.tif')
mask_sum = rxr.open_rasterio('data/GAEZ_v4/province_mask.tif')
mask_province = [(mask_sum == idx).expand_dims({'Province': [p]}) * mask
                    for idx, p in enumerate(UNIQUE_VALUES['Province'])]
mask_province = xr.combine_by_coords(mask_province)


# Create a zero array to pad the first year of the delta data
zero_xr = xr.DataArray(0, dims=['year'], coords={'year': [BASE_YR]})

# Load the pred_to_yearbook_ratio 
pred_to_yearbook_ratio = xr.open_dataset('data/results/step_16_GAEZ_yb_base_year_production.nc', chunks='auto')['data']



# Crop yield - yearbook trend (multiplier)
yield_MC_yrbook_trend = xr.open_dataset('data/results/step_7_Yearbook_MC_ratio.nc', chunks='auto')['data']

# Crop yield - climate change (t/ha)
yield_MC_attainable_trend = xr.open_dataset('data/results/step_7_GAEZ_yield_MC.nc', chunks='auto')['data']          

# Crop area - urban encroachment (km2)
area_start = xr.open_dataset('data/results/step_15_GAEZ_area_km2_adjusted.nc', chunks='auto')['data']
area_start = area_start * mask_province
area_crop_water_ratio = area_start.sum(dim=['y', 'x', 'band']) / area_start.sum(dim=['crop', 'water_supply', 'y', 'x', 'band'])
area_crop_water_ratio = area_crop_water_ratio * mask_province       # Assign the multiplier to each pixel in the given Province
area_urban_reduce = xr.open_dataset('data/results/step_13_urban_occupy_cropland.nc', chunks='auto')['data'] 
area_urban_reduce = area_urban_reduce * mask_province
area_urban_reduce = area_urban_reduce * area_crop_water_ratio

# Crop area - reclamation (km2)
area_reclaim_increase = xr.open_dataset('data/results/step_14_reclimation_area_km2.nc', chunks='auto')['data']
area_reclaim_increase = area_reclaim_increase * area_crop_water_ratio




# Get the delta for each impact factor
delta_yb = yield_MC_yrbook_trend.diff('year')
delta_yb = xr.concat([zero_xr, delta_yb], dim='year')

delta_climate = yield_MC_attainable_trend.diff('year')
delta_climate = xr.concat([zero_xr, delta_climate], dim='year')

delta_urban = area_urban_reduce.diff('year')
delta_urban = xr.concat([zero_xr, delta_urban], dim='year')

delta_reclaim = area_reclaim_increase.diff('year')
delta_reclaim = xr.concat([zero_xr, delta_reclaim], dim='year')


# Calculate the relative contribution of each impact factor to the production
contr_yb = (delta_yb * yield_MC_attainable_trend) \
         * (area_start + area_reclaim_increase - area_urban_reduce) \
         * 100 / 1e6 \
         * pred_to_yearbook_ratio

contr_climate = (delta_climate * yield_MC_attainable_trend) \
              * (area_start + area_reclaim_increase - area_urban_reduce) \
              * 100 / 1e6 \
              * pred_to_yearbook_ratio
              
contr_urban = (delta_urban * yield_MC_attainable_trend) \
            * (area_start + area_reclaim_increase - area_urban_reduce) \
            * 100 / 1e6 \
            * pred_to_yearbook_ratio
            
contr_reclaim = (delta_reclaim * yield_MC_attainable_trend) \
                * (area_start + area_reclaim_increase - area_urban_reduce) \
                * 100 / 1e6 \
                * pred_to_yearbook_ratio


if __name__ == '__main__':
    
    sel_dict = dict(
        sample=0, 
        crop='Wheat', 
        water_supply='Irrigated',
        rcp='RCP4.5',
        c02_fertilization='With CO2 Fertilization',
        SSP='SSP2')
    
    # Contribution from yearbook trend
    contr_yb_one = contr_yb.sel(**sel_dict, drop=True)
    contr_yb_one_stats = contr_yb_one.sum(['x', 'y']).compute().to_dataframe(name='delta').reset_index()
    
    fig_yb = (plotnine.ggplot(contr_yb_one_stats) +
            plotnine.aes(x='year', y='delta', color='Province') +
            plotnine.geom_line() +
            plotnine.theme_bw()
            )
    
    # Contribution from climate change
    contr_climate_one = contr_climate.sel(**sel_dict, drop=True)
    contr_climate_one_stats = contr_climate_one.sum(['x', 'y']).compute().to_dataframe(name='delta').reset_index()
    
    fig_climate = (plotnine.ggplot(contr_climate_one_stats) +
            plotnine.aes(x='year', y='delta', color='Province') +
            plotnine.geom_line() +
            plotnine.theme_bw()
            )
    
    # Contribution from urban encroachment
    contr_urban_one = contr_urban.sel(**sel_dict, drop=True)
    contr_urban_one_stats = contr_urban_one.sum(['x', 'y']).compute().to_dataframe(name='delta').reset_index()
    
    fig_urban = (plotnine.ggplot(contr_urban_one_stats) +
            plotnine.aes(x='year', y='delta', color='Province') +
            plotnine.geom_line() +
            plotnine.theme_bw()
            )
    
    # Contribution from reclamation
    contr_reclaim_one = contr_reclaim.sel(**sel_dict, drop=True)
    contr_reclaim_one_stats = contr_reclaim_one.sum(['x', 'y']).compute().to_dataframe(name='delta').reset_index()
    
    fig_reclaim = (plotnine.ggplot(contr_reclaim_one_stats) +
            plotnine.aes(x='year', y='delta', color='Province') +
            plotnine.geom_line() +
            plotnine.theme_bw()
            )




