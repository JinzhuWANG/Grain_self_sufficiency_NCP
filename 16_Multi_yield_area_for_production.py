import math
import xarray as xr
import rioxarray as rxr
import pandas as pd
import plotnine

from helper_func.calculate_GAEZ_stats import bincount_with_mask
from helper_func.parameters import UNIQUE_VALUES, Monte_Carlo_num

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
area_crop_water_ratio = area_crop_water_ratio * mask_province       # Assing the multiplier to each pixel in the given Province


area_urban_reduce = xr.open_dataset('data/results/step_13_urban_occupy_cropland.nc', chunks='auto')['data'] # km2
area_urban_reduce = area_urban_reduce * area_crop_water_ratio
area_reclaim_increase = xr.open_dataset('data/results/step_14_reclimation_area_km2.nc', chunks='auto')['data'] # km2
area_reclaim_increase = area_reclaim_increase * area_crop_water_ratio

area_pred = area_start + area_reclaim_increase - area_urban_reduce
area_pred = area_pred.sum(dim=['Province'])


# Get production
production_pred = yield_pred * area_pred * 100 / 1e6                      # million tonnes


if __name__ == '__main__':
    
    plotnine.options.figure_size = (18, 6)
    plotnine.options.dpi = 100
    

    production_pred_mean = production_pred.mean(dim=['sample'])
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
    production_stats = production_stats.groupby(['rcp', 'ssp', 'year', 'Province', 'crop','c02_fertilization']).sum().reset_index()
    
    production_stats['Province'] = production_stats['Province'].map(dict(enumerate(UNIQUE_VALUES['Province'])))
    production_stats['lower'] = production_stats['Production (Mt)'] - (production_stats['Production_std (Mt)'] / math.sqrt(Monte_Carlo_num)) * 1.96
    production_stats['upper'] = production_stats['Production (Mt)'] + (production_stats['Production_std (Mt)'] / math.sqrt(Monte_Carlo_num)) * 1.96
    
    # Filter data for plotting
    rcp = 'RCP2.6'
    ssp = 'SSP2'
    
    plot_df = production_stats.query(f'rcp == "{rcp}" & ssp == "{ssp}"').copy()
    
    g = (
        plotnine.ggplot() +
        plotnine.geom_line(
            plot_df, 
            plotnine.aes(x='year', y='Production (Mt)', color='c02_fertilization')) +
        plotnine.geom_ribbon(
            plot_df, 
            plotnine.aes(x='year', ymin='lower', ymax='upper', fill='c02_fertilization'), alpha=0.3) +
        plotnine.facet_grid('crop~Province', scales='free_y') +
        plotnine.theme_bw()
    )
