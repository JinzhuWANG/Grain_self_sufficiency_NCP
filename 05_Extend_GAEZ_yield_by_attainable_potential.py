import math
from matplotlib import axis
import rioxarray as rxr
import xarray as xr
import numpy as np
import plotnine

from helper_func.calculate_GAEZ_stats import bincount_with_mask
from helper_func.get_yearbook_records import get_yearbook_yield
from helper_func.parameters import UNIQUE_VALUES


# Load the data
GAEZ_attain_mean = xr.open_dataset("data/results/step_3_GAEZ_AY_GYGA_mean.nc")['data']        # t/ha
GAEZ_attain_std = xr.open_dataset("data/results/step_3_GAEZ_AY_GYGA_std.nc")['data']/1000     # kg/ha -> t/ha

# Divide the start year to get ratio change
GAEZ_attain_mean_ratio = GAEZ_attain_mean / GAEZ_attain_mean.sel(year=2020)
GAEZ_attain_mean_ratio = GAEZ_attain_mean_ratio.where(~np.isinf(GAEZ_attain_mean_ratio), 0)

# Assume the yield will not decrease
GAEZ_attain_mean_ratio = GAEZ_attain_mean_ratio.where(GAEZ_attain_mean_ratio > 1, 1).sel(year=slice(2020, 2101))
GAEZ_attain_mean_ratio = GAEZ_attain_mean_ratio.where(GAEZ_attain_mean_ratio < 3, 3)    # Any yield increase more than 3 times will be capped at 3

# Load the actual yield data
GAEZ_actual_yield = xr.open_dataset("data/results/step_4_GAEZ_actual_yield_adj.nc")['data']

# Extend the yield by multiplying the ratio
GAEZ_actual_yield = GAEZ_actual_yield * GAEZ_attain_mean_ratio

encoding = {'data': {'dtype': 'float32', 'zlib': True}}
GAEZ_actual_yield.to_netcdf('data/results/step_5_GAEZ_actual_yield_extended.nc', encoding=encoding, engine='h5netcdf')




if __name__ == '__main__':

    # yearbook yield
    yearbook_yield = get_yearbook_yield().query('year >= 1990')
    
    # Read masks
    mask_sum = rxr.open_rasterio('data/GAEZ_v4/Province_mask.tif')
    mask_mean = rxr.open_rasterio('data/GAEZ_v4/Province_mask_mean.tif')

    yield_mean_stats = bincount_with_mask(mask_sum, GAEZ_actual_yield * mask_mean)
    yield_mean_stats = yield_mean_stats.rename(columns={'Value': 'yield', 'bin': 'Province'})
    
    yield_std_stats = bincount_with_mask(mask_sum, GAEZ_attain_std * mask_mean )           
    yield_std_stats = yield_std_stats.rename(columns={'Value': 'yield_std', 'bin': 'Province'})

    yield_stats = yield_mean_stats.merge(yield_std_stats, how='left')
    yield_stats['Province'] = yield_stats['Province'].map(dict(enumerate(UNIQUE_VALUES['Province'])))
    yield_stats = yield_stats.sort_values(['rcp', 'crop', 'water_supply', 'c02_fertilization', 'Province', 'year'])
    
    yield_stats['upper'] = yield_stats.groupby(['rcp', 'crop', 'water_supply', 'c02_fertilization', 'Province'])[['yield','yield_std']]\
        .apply(lambda x: x['yield'] + (1.96 * x['yield_std'] / math.sqrt(len(x)))).values
    yield_stats['lower'] = yield_stats.groupby(['rcp', 'crop', 'water_supply', 'c02_fertilization', 'Province'])[['yield','yield_std']]\
        .apply(lambda x: x['yield'] - (1.96 * x['yield_std'] / math.sqrt(len(x)))).values
    
    
    
    
    # Plot the yield
    rcp='RCP2.6'
    c02_fert = 'With CO2 Fertilization'
    yield_stats = yield_stats.query(f'rcp == "{rcp}" & c02_fertilization == "{c02_fert}"')

    plotnine.options.figure_size = (16, 6)
    plotnine.options.dpi = 100

    g = (plotnine.ggplot() +
        plotnine.geom_point(yearbook_yield, plotnine.aes(x='year', y='Yield (tonnes)'), size=0.05, alpha=0.5) +
        plotnine.geom_line(yield_stats, plotnine.aes(x='year', y='yield', color='water_supply')) +
        plotnine.geom_ribbon(yield_stats, plotnine.aes(x='year', ymin='lower', ymax='upper', fill='water_supply'), alpha=0.4) +
        plotnine.labs(x='Year', y='Yield (t/ha)') +
        plotnine.facet_grid('crop~Province', scales='free_y') +
        plotnine.theme_bw() 
        )

    g.save('data/results/fig_step_5_GAEZ_actual_yield_extrapolate_by_attainable_multiplier_RCP26_t_ha.svg')







