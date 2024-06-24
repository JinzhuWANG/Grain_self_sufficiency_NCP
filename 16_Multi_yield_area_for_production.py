from calendar import c
import math
import xarray as xr
import rioxarray as rxr
import pandas as pd
import plotnine

from helper_func.calculate_GAEZ_stats import bincount_with_mask, get_GAEZ_stats
from helper_func.get_yearbook_records import get_yearbook_area, get_yearbook_production
from helper_func.parameters import UNIQUE_VALUES, Monte_Carlo_num

# Get masks
mask = rxr.open_rasterio('data/GAEZ_v4/GAEZ_mask.tif')
mask_mean = rxr.open_rasterio('data/GAEZ_v4/Province_mask_mean.tif')
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
area_crop_water_ratio = area_crop_water_ratio * mask_province       # Assing the multiplier to each pixel in the given Province
area_urban_reduce = xr.open_dataset('data/results/step_13_urban_occupy_cropland.nc', chunks='auto')['data'] # 
area_urban_reduce = area_urban_reduce * mask_province
area_urban_reduce = area_urban_reduce * area_crop_water_ratio
area_reclaim_increase = xr.open_dataset('data/results/step_14_reclimation_area_km2.nc', chunks='auto')['data'] # km2
area_reclaim_increase = area_reclaim_increase * area_crop_water_ratio

# Get production
area_pred = area_start + area_reclaim_increase - area_urban_reduce
area_pred = area_pred.sum('Province')
production_pred = yield_pred * area_pred * 100 / 1e6                      # million tonnes


# Calculate the difference between the yearbook and the predicted production
production_pred_mean = production_pred.mean(dim=['sample'])
production_mean_stats = bincount_with_mask(mask_sum, production_pred_mean)
production_mean_stats = production_mean_stats.rename(
    columns={'Value': 'Production (Mt)', 'bin': 'Province'})
production_mean_stats['Province'] = production_mean_stats['Province'].map(dict(enumerate(UNIQUE_VALUES['Province'])))

production_pred_start = production_mean_stats\
    .query('year == 2020 & rcp=="RCP2.6" & ssp =="SSP2" & c02_fertilization=="With CO2 Fertilization"').copy()
production_pred_start = production_pred_start\
    .groupby(['Province','crop'])\
    .sum(numeric_only=True).reset_index()
    
yearbook_production = get_yearbook_production().query('year >= 1990')
yearbook_production_start = yearbook_production.query('year == 2020').copy()

GAEZ_yr_production = pd.merge(
    production_pred_start,
    yearbook_production_start,
    on = ['Province', 'crop'],
    suffixes = ('_pred', '_yrbook')
)
GAEZ_yr_production['pred_to_yrbook_ratio'] = GAEZ_yr_production['Production (Mt)_yrbook'] / GAEZ_yr_production['Production (Mt)_pred'] 

GAEZ_yr_production_xr = xr.Dataset.from_dataframe(
    GAEZ_yr_production.set_index(['Province', 'crop'])[['pred_to_yrbook_ratio']])     
GAEZ_yr_production_xr = GAEZ_yr_production_xr['pred_to_yrbook_ratio'] * mask_province
GAEZ_yr_production_xr = GAEZ_yr_production_xr.sum(dim=['Province'])

# Forcing the production in the starting year to be the same as the yearbook
production_pred_mean_adj = production_pred * GAEZ_yr_production_xr




if __name__ == '__main__':
    
    plotnine.options.figure_size = (10, 8)
    plotnine.options.dpi = 100
    
    yearbook_production_sum = yearbook_production\
        .groupby(['year']).sum(numeric_only=True).reset_index()


    production_pred_mean = production_pred_mean_adj.mean(dim=['sample'])
    production_pred_std = production_pred.std(dim=['sample'])

    production_mean_stats = bincount_with_mask(mask_sum, production_pred_mean)
    production_std_stats = bincount_with_mask(mask_sum, production_pred_std)
    
    production_stats = pd.merge(
        production_mean_stats, 
        production_std_stats, 
        on = [ 'rcp', 'ssp', 'year', 'bin', 'crop', 'water_supply', 'c02_fertilization'], 
        suffixes = ('_mean', '_std'))
    production_stats = production_stats.rename(
        columns = {'Value_mean': 'Production (Mt)', 'Value_std': 'Production_std (Mt)', 'bin': 'Province'})
    production_stats = production_stats.groupby(['rcp', 'ssp', 'year', 'Province', 'crop', 'c02_fertilization']).sum().reset_index()
    
    production_stats['Province'] = production_stats['Province'].map(dict(enumerate(UNIQUE_VALUES['Province'])))
    production_stats['se'] = production_stats['Production_std (Mt)'] / math.sqrt(Monte_Carlo_num)
    production_stats['lower'] = production_stats['Production (Mt)'] - production_stats['se'] * 1.96
    production_stats['upper'] = production_stats['Production (Mt)'] + production_stats['se'] * 1.96
    
    # Filter data for plotting
    rcp = 'RCP2.6'
    co2 = 'With CO2 Fertilization'

    plot_df = production_stats.query(
        f'rcp == "{rcp}"  & c02_fertilization == "{co2}"').copy()
    plot_df_sum = plot_df.groupby(['year','ssp']).sum(numeric_only=True).reset_index()
    
   
    g = (
        plotnine.ggplot() +
        plotnine.geom_point(
            yearbook_production_sum, 
            plotnine.aes(x='year', y='Production (Mt)'), color='grey', size=0.5) +
        plotnine.geom_line(
            plot_df_sum, 
            plotnine.aes(x='year', y='Production (Mt)', color='ssp')) +
        plotnine.geom_ribbon(
            plot_df_sum, 
            plotnine.aes(x='year', ymin='lower', ymax='upper', fill='ssp'), alpha=0.3) +
        plotnine.theme_bw()
    )

    g.save('data/results/fig_step_16_Pred_total_grain_production.svg')