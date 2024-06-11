import math
import pandas as pd
import plotnine
import xarray as xr
import rioxarray as rxr

from helper_func import sample_ppf
from helper_func.calculate_GAEZ_stats import bincount_with_mask
from helper_func.parameters import (BASE_YR, 
                                    TARGET_YR, 
                                    PRED_STEP,
                                    UNIQUE_VALUES, 
                                    Monte_Carlo_num)



# Define parameters
chunk_size = {
    'crop': 3,
    'water_supply': 2,
    'y': 160,
    'x': 149,
    'band': 1,
    'year': 1,
    'rcp': 4,
    'c02_fertilization': 2
}


# Read data
yearbook_trend = xr.open_dataset('data/results/step_6_yearbook_yield_extrapolated.nc')

mask_sum = rxr.open_rasterio('data/GAEZ_v4/Province_mask.tif')
mask_mean = rxr.open_rasterio('data/GAEZ_v4/Province_mask_mean.tif')
mask_province = [xr.where(mask_sum == idx, 1, 0).expand_dims({'Province': [p]}) 
                 for idx,p in enumerate(UNIQUE_VALUES['Province'])]
mask_province = xr.combine_by_coords(mask_province).astype('float32') 

GAEZ_attainable_yield = xr.open_dataset('data/results/step_3_GAEZ_AY_GYGA_mean.nc', chunks='auto')['data']/1000     # kg/ha -> t/ha
GAEZ_attainable_yield = GAEZ_attainable_yield.sel(year=slice(BASE_YR, TARGET_YR + 1))

GAEZ_yield_mean = xr.open_dataset('data/results/step_5_GAEZ_actual_yield_extended.nc')                       # t/ha
GAEZ_yield_mean = GAEZ_yield_mean['data'].sel(year=slice(BASE_YR, TARGET_YR + 1)).astype('float32').chunk(chunk_size)

GAEZ_yield_std = xr.open_dataset('data/results/step_3_GAEZ_AY_GYGA_std.nc') / 1000                           # kg/ha -> t/ha
GAEZ_yield_std = GAEZ_yield_std['data'].sel(year=slice(BASE_YR, TARGET_YR + 1))
GAEZ_yield_std = GAEZ_yield_std.transpose(*GAEZ_yield_mean.dims).astype('float32')
GAEZ_yield_std = GAEZ_yield_std.where(GAEZ_yield_std > 0, 1e-3).chunk(chunk_size)


# Sample from GAEZ mean and std
GAEZ_MC = sample_ppf(GAEZ_yield_mean, GAEZ_yield_std, n_samples=Monte_Carlo_num)
GAEZ_MC = xr.DataArray(
    GAEZ_MC, 
    dims=('sample',) + GAEZ_yield_mean.dims, 
    coords={'sample':range(Monte_Carlo_num), **GAEZ_yield_mean.coords}
    )


# Sample from yearbook trend
Yearbook_MC = sample_ppf(yearbook_trend['mean'], yearbook_trend['std'], n_samples=Monte_Carlo_num)
Yearbook_MC = xr.DataArray(
    Yearbook_MC, 
    dims=('sample',) + yearbook_trend['mean'].dims, 
    coords={'sample':range(Monte_Carlo_num), **yearbook_trend['mean'].coords}
    )


# Assign ratio value to mask, so all pixels inside a province have the same value
Yearbook_MC = Yearbook_MC * mask_province
Yearbook_MC = Yearbook_MC.sum(dim='Province') # Mosaic across province to get full map

Yearbook_MC_ratio = Yearbook_MC / Yearbook_MC.sel(year=2020)


# Get the year when yield touches the attainable ceiling
Yield_MC = GAEZ_MC * Yearbook_MC_ratio
Yield_MC_mean = Yield_MC.mean(dim='sample')

Yield_MC_stats = bincount_with_mask(mask_sum, Yield_MC_mean * mask_mean)
Attainable_stats = bincount_with_mask(mask_sum, GAEZ_attainable_yield * mask_mean)

Yield_with_Attain = Yield_MC_stats.merge(
    Attainable_stats, 
    on=['bin', 'crop', 'water_supply', 'year', 'rcp', 'c02_fertilization'],
    suffixes=('_yield', '_attainable'))

Yield_with_Attain['diff'] = abs(Yield_with_Attain['Value_yield'] - Yield_with_Attain['Value_attainable'])

min_diff_yr = pd.DataFrame()
for idx, df in Yield_with_Attain.groupby(
    ['crop', 'water_supply', 'rcp', 'c02_fertilization']):
    min_yr = pd.DataFrame([{
        'crop': idx[0],
        'water_supply': idx[1],
        'rcp': idx[2],
        'c02_fertilization': idx[3],
        'year':df.loc[df['diff'].idxmin()]['year']}])
    min_diff_yr = pd.concat([min_diff_yr,min_yr])


# The yearbook trend will stop at the year when the yield touches the attainable ceiling
ceilling_splits = []
for idx, row in min_diff_yr.iterrows():
    sel_xr = Yearbook_MC_ratio.sel(crop=row['crop']).expand_dims('crop')
    if row['year'] >= TARGET_YR:
        ceilling_splits.append(sel_xr)
        continue
    before_ceiling = sel_xr.sel(year=slice(BASE_YR, row['year']))
    after_ceiling = sel_xr.sel(year=row['year'] + PRED_STEP).expand_dims({'year': range(row['year'] + 5, TARGET_YR + 1, 5)})
    ceilling_splits.append(xr.combine_by_coords([before_ceiling, after_ceiling]))

Yearbook_MC_ratio_ceiling = xr.combine_by_coords(ceilling_splits).astype('float32')


# Save the data
GAEZ_MC.name = 'data'
Yearbook_MC_ratio_ceiling.name = 'data'

encoding = {'data': {'zlib': True, 'dtype': 'float32', 'complevel': 9}}
GAEZ_MC.to_netcdf('data/results/step_7_GAEZ_yield_MC.nc', encoding=encoding, engine='h5netcdf')
Yearbook_MC_ratio_ceiling.to_netcdf('data/results/step_7_Yearbook_MC_ratio.nc', encoding=encoding, engine='h5netcdf')








if __name__ == '__main__':
    
    plotnine.options.figure_size = (16, 6)
    plotnine.options.dpi = 100
    
    # Filter the data with rcp and c02_fertilization
    rcp = 'RCP2.6'
    co2 = 'With CO2 Fertilization'
    
    
    # Check the difference between GAEZ and GAEZ_MC
    GAEZ_MC_mean = GAEZ_MC.mean(dim='sample').compute()
    GAEZ_MC_std = GAEZ_MC.std(dim='sample').compute()
    
    yield_mean_stats = bincount_with_mask(mask_sum, GAEZ_MC_mean * mask_mean)
    yield_std_stats = bincount_with_mask(mask_sum, GAEZ_MC_std * mask_mean)      
    
    yield_stats = yield_mean_stats.merge(
        yield_std_stats, 
        on = ['bin','crop', 'water_supply', 'year', 'rcp', 'c02_fertilization'], 
        suffixes = ('_mean', '_std')
    )
    
    yield_stats = yield_stats.rename(columns={ 'bin': 'Province', 'Value_std': 'yield_std', 'Value_mean': 'yield_mean'})
    yield_stats['Province'] = yield_stats['Province'].map(dict(enumerate(UNIQUE_VALUES['Province'])))
    
    plot_df = yield_stats.query(f"rcp == '{rcp}' & c02_fertilization == '{co2}'").copy()
    plot_df['upper'] = plot_df['yield_mean'] + (plot_df['yield_std'] / math.sqrt(Monte_Carlo_num) * 1.96)
    plot_df['lower'] = plot_df['yield_mean'] - (plot_df['yield_std'] / math.sqrt(Monte_Carlo_num) * 1.96)

    g = (plotnine.ggplot() +
        plotnine.geom_line(plot_df, plotnine.aes(x='year', y='yield_mean', color='water_supply')) +
        plotnine.geom_ribbon(plot_df, plotnine.aes(x='year', ymin='lower', ymax='upper', fill='water_supply'), alpha=0.4) +
        plotnine.labs(x='Year', y='Yield (t/ha)') +
        plotnine.facet_grid('crop~Province', scales='free_y') +
        plotnine.theme_bw() 
        )
    
    
    
    # Multiply the GAEZ_MC by the Yearbook_MC_ratio
    GAEZ_yb = GAEZ_MC * Yearbook_MC_ratio_ceiling
    GAEZ_yb_mean = GAEZ_yb.mean(dim='sample').compute()
    GAEZ_yb_std = GAEZ_yb.std(dim='sample').compute()
    
    
    GAEZ_yb_mean_stats = bincount_with_mask(mask_sum, GAEZ_yb_mean * mask_mean)
    GAEZ_yb_std_stats = bincount_with_mask(mask_sum, GAEZ_yb_std * mask_mean)
    
    GAEZ_yb_stats = GAEZ_yb_mean_stats.merge(
        GAEZ_yb_std_stats,
        on = ['bin', 'crop', 'water_supply', 'year', 'rcp', 'c02_fertilization'], 
        suffixes = ('_mean', '_std'))
    
    GAEZ_yb_stats = GAEZ_yb_stats.rename(
        columns = {
            'bin': 'Province', 
            'Value_mean': 'Yield_mean (t/ha)', 
            'Value_std': 'Yield_std (t/ha)'
            })
    
    GAEZ_yb_stats['Province'] = GAEZ_yb_stats['Province'].map(dict(enumerate(UNIQUE_VALUES['Province'])))
    
    
    plot_df = GAEZ_yb_stats.query(f"rcp == '{rcp}' & c02_fertilization == '{co2}'").copy()
    plot_df['upper'] = plot_df['Yield_mean (t/ha)'] + (plot_df['Yield_std (t/ha)'] / math.sqrt(5) * 1.96)
    plot_df['lower'] = plot_df['Yield_mean (t/ha)'] - (plot_df['Yield_std (t/ha)'] / math.sqrt(5) * 1.96)
        
    g = (plotnine.ggplot(plot_df) +
         plotnine.geom_line(plotnine.aes(x = 'year', y = 'Yield_mean (t/ha)', color = 'water_supply')) +
         plotnine.geom_ribbon(plotnine.aes(x = 'year', ymin = 'lower', ymax = 'upper', fill = 'water_supply'), alpha = 0.2) +
         plotnine.facet_grid('crop~Province') +
         plotnine.theme_bw()
        )

