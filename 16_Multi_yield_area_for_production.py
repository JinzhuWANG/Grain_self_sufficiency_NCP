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



# Get production prediction under different SSPs
population_hist = pd.read_csv('data/results/POP_NCP.csv')
population_hist['SSP'] = 'Historical'
population_hist['Population_million'] = population_hist['Value'] / 100
population_hist_sum = population_hist.groupby(['year', 'SSP']).sum(numeric_only=True).reset_index()

population_pred = pd.read_csv('data/results/POP_NCP_pred.csv')
population_pred['Population_million'] = population_pred['Value'] / 100

population = pd.concat([population_hist, population_pred], ignore_index=True)
population_sum = population.groupby(['year', 'SSP']).sum(numeric_only=True).reset_index()
population_sum_province = population.groupby(['year', 'SSP', 'Province']).sum(numeric_only=True).reset_index()

# Get yearbook production
yearbook_production = get_yearbook_production().query('year >= 1990')
yearbook_production_sum = yearbook_production\
    .groupby(['year']).sum(numeric_only=True).reset_index()
yearbook_production_hist_sum = pd.merge(
    yearbook_production_sum,
    population_hist_sum,
    on = ['year']
)
yearbook_production_hist_sum['Production_per_capita_kg'] = \
    yearbook_production_hist_sum.eval('`Production (Mt)` / Population_million') * 1e3


yearbook_production_province = yearbook_production\
    .groupby(['year', 'Province']).sum(numeric_only=True).reset_index()
yearbook_production_hist_per_capita = pd.merge(
    yearbook_production_province,
    population_sum_province,
    on=['year', 'Province']
)
yearbook_production_hist_per_capita['Production_per_capita_kg'] = \
    yearbook_production_hist_per_capita.eval('`Production (Mt)` / Population_million') * 1e3


# Crop yield (t/ha)
yield_MC_yrbook_trend = xr.open_dataset('data/results/step_7_Yearbook_MC_ratio.nc', chunks='auto')['data']          
yield_MC_attainable_trend = xr.open_dataset('data/results/step_7_GAEZ_yield_MC.nc', chunks='auto')['data']          # t/ha
yield_pred = yield_MC_attainable_trend * yield_MC_yrbook_trend


# Urban encroachment (km2)
'''
Because the urban encroachment area is a total number for each province, we need to seperate it for 
each crop and water_supply.

For each province, we calculate the area ratio of each crop and water_supply to the total cropland area, 
then use this ratio as weight to seperate the total cropland loss (i.e. urban encroachment) to each crop and water_supply.
'''
area_start = xr.open_dataset('data/results/step_15_GAEZ_area_km2_adjusted.nc', chunks='auto')['data']   # km2
area_start = area_start * mask_province

area_crop_water_ratio = area_start.sum(dim=['y', 'x', 'band']) / area_start.sum(dim=['crop', 'water_supply', 'y', 'x', 'band'])
area_crop_water_ratio = area_crop_water_ratio * mask_province       # Assign the multiplier to each pixel in the given Province
area_urban_reduce = xr.open_dataset('data/results/step_13_urban_occupy_cropland.nc', chunks='auto')['data'] # km2
area_urban_reduce = area_urban_reduce * mask_province
area_urban_reduce = area_urban_reduce * area_crop_water_ratio

# Reclamation area (km2)
area_reclaim_increase = xr.open_dataset('data/results/step_14_reclimation_area_km2.nc', chunks='auto')['data'] # km2
area_reclaim_increase = area_reclaim_increase * area_crop_water_ratio


# Get production
area_pred = area_start + area_reclaim_increase - area_urban_reduce
area_pred = area_pred.sum('Province')
production_pred = yield_pred * (area_pred * 100) / 1e6                      # million tonnes


# Calculate the difference between the yearbook and the predicted production
production_pred_mean = production_pred.mean(dim=['sample'])

production_mean_stats = bincount_with_mask(mask_sum, production_pred_mean)
production_mean_stats = production_mean_stats.rename(columns={'Value': 'Production (Mt)', 'bin': 'Province'})
production_mean_stats['Province'] = production_mean_stats['Province'].map(dict(enumerate(UNIQUE_VALUES['Province'])))
production_pred_start = production_mean_stats\
    .query('year == 2020 & rcp=="RCP2.6" & SSP =="SSP2" & c02_fertilization=="With CO2 Fertilization"')\
    .copy()\
    .groupby(['Province','crop'])\
    .sum(numeric_only=True)\
    .reset_index()
    
yearbook_production_start = get_yearbook_production().query('year == 2020').copy()

GAEZ_yr_production = pd.merge(
    production_pred_start,
    yearbook_production_start,
    on = ['Province', 'crop'],
    suffixes = ('_pred', '_yrbook')
)

GAEZ_yr_production['pred_to_yrbook_ratio'] = GAEZ_yr_production['Production (Mt)_yrbook'] / GAEZ_yr_production['Production (Mt)_pred'] 
GAEZ_yr_production.to_csv('data/results/step_16_GAEZ_yb_base_year_production.csv', index=False)

# Convert the pred_to_yrbook_ratio to xarray
GAEZ_yr_production_xr = xr.Dataset.from_dataframe(
    GAEZ_yr_production.set_index(['Province', 'crop'])[['pred_to_yrbook_ratio']])     
GAEZ_yr_production_xr = GAEZ_yr_production_xr['pred_to_yrbook_ratio'] * mask_province
GAEZ_yr_production_xr = GAEZ_yr_production_xr.sum(dim=['Province'])


# Forcing the production in the starting year to be the same as the yearbook
encoding = {'data': {'dtype': 'float32', 'zlib': True, 'complevel': 9}}
GAEZ_yr_production_xr.name = 'data'
GAEZ_yr_production_xr.to_netcdf('data/results/step_16_GAEZ_yb_base_year_production.nc', encoding=encoding)

production_pred_mean_adj = production_pred * GAEZ_yr_production_xr
production_pred_mean_adj.name = 'data'
production_pred_mean_adj.to_netcdf('data/results/step_16_production_pred_adj.nc', encoding=encoding)




# Calculate the stats from the adjusted production
production_pred_mean = production_pred_mean_adj.mean(dim=['sample'])
production_pred_std = production_pred_mean_adj.std(dim=['sample'])

production_mean_stats = bincount_with_mask(mask_sum, production_pred_mean)
production_std_stats = bincount_with_mask(mask_sum, production_pred_std)

production_stats = pd.merge(
    production_mean_stats, 
    production_std_stats, 
    on = [ 'rcp', 'SSP', 'year', 'bin', 'crop', 'water_supply', 'c02_fertilization'], 
    suffixes = ('_mean', '_std'))
production_stats = production_stats.rename(
    columns = {
        'Value_mean': 'Production_total_Mt', 
        'Value_std': 'Production_std_total_Mt', 
        'bin': 'Province'
    })
production_stats = production_stats.groupby(['rcp', 'SSP', 'year', 'water_supply', 'Province', 'crop', 'c02_fertilization']).sum().reset_index()
production_stats['Province'] = production_stats['Province'].map(dict(enumerate(UNIQUE_VALUES['Province'])))
production_stats['se'] = production_stats['Production_std_total_Mt'] / math.sqrt(Monte_Carlo_num)
production_stats['lower_total_MT'] = production_stats['Production_total_Mt'] - production_stats['se'] * 1.96
production_stats['upper_total_MT'] = production_stats['Production_total_Mt'] + production_stats['se'] * 1.96

production_stats.to_csv('data/results/step_16_production_stats.csv', index=False)



if __name__ == '__main__':
    
    plotnine.options.figure_size = (10, 8)
    plotnine.options.dpi = 100
    
    production_stats = pd.read_csv('data/results/step_16_production_stats.csv')

    # Filter data for plotting
    rcp = 'RCP2.6'
    co2 = 'With CO2 Fertilization'

    plot_df = production_stats.query(f'rcp == "{rcp}"  & c02_fertilization == "{co2}"').copy()
    plot_df_sum = plot_df.groupby(['year','SSP']).sum(numeric_only=True).reset_index()
    plot_df_sum_province = plot_df.groupby(['year','SSP', 'Province']).sum(numeric_only=True).reset_index()
    
    # Total grain production
    g = (
        plotnine.ggplot() +
        plotnine.geom_point(
            yearbook_production_sum, 
            plotnine.aes(x='year', y='Production (Mt)'), color='grey', size=0.5) +
        plotnine.geom_line(
            plot_df_sum, 
            plotnine.aes(x='year', y='Production_total_Mt', color='SSP')) +
        plotnine.geom_ribbon(
            plot_df_sum, 
            plotnine.aes(x='year', ymin='lower_total_MT', ymax='upper_total_MT', fill='SSP'), alpha=0.3) +
        plotnine.theme_bw()
    )

    g.save('data/results/fig_step_16_Pred_total_grain_production.svg')
    
    
    # Total grain production for each province
    plot_df_sum_province = plot_df.groupby(['year','SSP', 'Province']).sum(numeric_only=True).reset_index()
    
    g = (
        plotnine.ggplot() +
        plotnine.geom_point(
            yearbook_production_province, 
            plotnine.aes(x='year', y='Production (Mt)'), color='grey', size=0.5) +
        plotnine.geom_line(
            plot_df_sum_province, 
            plotnine.aes(x='year', y='Production_total_Mt', color='SSP')) +
        plotnine.geom_ribbon(
            plot_df_sum_province, 
            plotnine.aes(x='year', ymin='lower_total_MT', ymax='upper_total_MT', fill='SSP'), alpha=0.3) +
        plotnine.theme_bw() +
        plotnine.facet_wrap('~Province', scales='free_y')
    )
    g.save('data/results/fig_step_16_Pred_total_grain_production_province.svg')
    
    
    # Per capita grain production
    plot_df_total_per_capita = pd.merge(
        plot_df_sum,
        population_sum,
        on = ['year', 'SSP'],
        suffixes = ('', '_pop')
    )
    plot_df_total_per_capita['Production_per_capita_mean_kg'] = \
        plot_df_total_per_capita.eval('Production_total_Mt / Population_million') * 1000
    plot_df_total_per_capita['Production_per_capita_lower_kg'] = \
        plot_df_total_per_capita.eval('lower_total_MT / Population_million') * 1000
    plot_df_total_per_capita['Production_per_capita_upper_kg'] = \
        plot_df_total_per_capita.eval('upper_total_MT / Population_million') * 1000
        
    g = (
        plotnine.ggplot() +
        plotnine.geom_point(
            yearbook_production_hist_sum, 
            plotnine.aes(x='year', y='Production_per_capita_kg'), color='grey', size=0.5) +
        plotnine.geom_line(
            plot_df_total_per_capita, 
            plotnine.aes(x='year', y='Production_per_capita_mean_kg', color='SSP')) +
        plotnine.geom_ribbon(
            plot_df_total_per_capita, 
            plotnine.aes(x='year', ymin='Production_per_capita_lower_kg', ymax='Production_per_capita_upper_kg', fill='SSP'), alpha=0.3) +
        plotnine.theme_bw()
    )
    g.save('data/results/fig_step_16_Pred_per_capita_grain_production.svg')
    
    
    # Per capita grain production for each province
    plot_df_per_capita_province = pd.merge(
        plot_df_sum_province,
        population_sum_province,
        on = ['year', 'SSP', 'Province'],
        suffixes = ('', '_pop')
    )
    
    plot_df_per_capita_province['Production_per_capita_mean_kg'] = \
        plot_df_per_capita_province.eval('Production_total_Mt / Population_million') * 1000
    plot_df_per_capita_province['Production_per_capita_lower_kg'] = \
        plot_df_per_capita_province.eval('lower_total_MT / Population_million') * 1000
    plot_df_per_capita_province['Production_per_capita_upper_kg'] = \
        plot_df_per_capita_province.eval('upper_total_MT / Population_million') * 1000
        
    g = (
        plotnine.ggplot() +
        plotnine.geom_point(
            yearbook_production_hist_per_capita, 
            plotnine.aes(x='year', y='Production_per_capita_kg'), color='grey', size=0.5) +
        plotnine.geom_line(
            plot_df_per_capita_province, 
            plotnine.aes(x='year', y='Production_per_capita_mean_kg', color='SSP')) +
        plotnine.geom_ribbon(
            plot_df_per_capita_province, 
            plotnine.aes(x='year', ymin='Production_per_capita_lower_kg', ymax='Production_per_capita_upper_kg', fill='SSP'), alpha=0.3) +
        plotnine.theme_bw() +
        plotnine.facet_wrap('~Province', scales='free_y')
    )
    g.save('data/results/fig_step_16_Pred_per_capita_grain_production_province.svg')

      
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    