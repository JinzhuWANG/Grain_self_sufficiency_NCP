import xarray as xr
import plotnine
import rioxarray as rxr

from helper_func import sample_ppf
from helper_func.calculate_GAEZ_stats import bincount_with_mask
from helper_func.parameters import UNIQUE_VALUES, Monte_Carlo_num



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
mask_province = [xr.where(mask_sum == idx, 1, 0).expand_dims({'Province': [p]}) for idx,p in enumerate(UNIQUE_VALUES['Province'])]
mask_province = xr.combine_by_coords(mask_province).astype('float32') 

GAEZ_yield_mean = xr.open_dataset('data/results/step_5_GAEZ_actual_yield_extended.nc')
GAEZ_yield_mean = GAEZ_yield_mean['data'].sel(year=slice(2020, 2101)).astype('float32').chunk(chunk_size)

GAEZ_yield_std = xr.open_dataset('data/results/step_3_GAEZ_AY_GYGA_std.nc') / 1000   # kg/ha to t/ha
GAEZ_yield_std = GAEZ_yield_std['data'].sel(year=slice(2020, 2101))
GAEZ_yield_std = GAEZ_yield_std.transpose(*GAEZ_yield_mean.dims).astype('float32')
GAEZ_yield_std = GAEZ_yield_std.where(GAEZ_yield_std > 0, 1e-3).chunk(chunk_size)


# Sample from GAEZ mean and std
GAEZ_MC = sample_ppf(GAEZ_yield_mean, GAEZ_yield_std, n_samples=Monte_Carlo_num)
GAEZ_MC = xr.DataArray(
    GAEZ_MC, 
    dims=('sample',) + GAEZ_yield_mean.dims, 
    coords={'sample':range(Monte_Carlo_num), **GAEZ_yield_mean.coords}
    )
GAEZ_MC.name = 'data'


# Sample from yearbook trend
Yearbook_MC = sample_ppf(yearbook_trend['mean'], yearbook_trend['std'], n_samples=Monte_Carlo_num)
Yearbook_MC = xr.DataArray(
    Yearbook_MC, 
    dims=('sample',) + yearbook_trend['mean'].dims, 
    coords={'sample':range(Monte_Carlo_num), **yearbook_trend['mean'].coords}
    )

# Multiply by the mask, so all pixels inside a province have the same value
Yearbook_MC = Yearbook_MC * mask_province
Yearbook_MC = Yearbook_MC.sum(dim='Province') # Mosaic across province to get full map

Yearbook_MC_ratio = Yearbook_MC / Yearbook_MC.sel(year=2020)
Yearbook_MC_ratio.name = 'data'


# Save the data
encoding = {'data': {'zlib': True, 'dtype': 'float32', 'complevel': 9}}
GAEZ_MC.to_netcdf('data/results/step_7_GAEZ_yield_MC.nc', encoding=encoding, engine='h5netcdf')
Yearbook_MC_ratio.to_netcdf('data/results/step_7_Yearbook_MC_ratio.nc', encoding=encoding, engine='h5netcdf')





if __name__ == '__main__':
    
    # Check the difference between GAEZ and GAEZ_MC
    GAEZ_mean = GAEZ_yield_mean.sel(year=2070)
    GAEZ_std = GAEZ_yield_std.sel(year=2070)
    GAEZ_MC_mean = GAEZ_MC.mean(dim='sample').sel(year=2070).compute()
    GAEZ_MC_std = GAEZ_MC.std(dim='sample').sel(year=2070).compute()
    
    diff_mean = GAEZ_mean - GAEZ_MC_mean  
    diff_std = GAEZ_MC_std - GAEZ_std 
    
    diff_mean[0,0,...,0,0,0].plot()
    diff_std[0,0,...,0,0,0].plot()
    
    
    # Multiply the GAEZ_MC by the Yearbook_MC_ratio
    GAEZ_yb = GAEZ_MC * Yearbook_MC_ratio
    GAEZ_yb_mean = GAEZ_yb.mean(dim='sample').compute()
    GAEZ_yb_std = GAEZ_yb.std(dim='sample').compute()
    
    mask_mean = rxr.open_rasterio('data/GAEZ_v4/Province_mask_mean.tif')
    GAEZ_yb_mean_stats = bincount_with_mask(mask_sum, GAEZ_yb_mean * mask_mean)
    GAEZ_yb_std_stats = bincount_with_mask(mask_sum, GAEZ_yb_std * mask_mean)
    
    GAEZ_yb_stats = GAEZ_yb_mean_stats.merge(GAEZ_yb_std_stats,
                                             on = ['bin', 'crop', 'water_supply', 'year', 'rcp', 'c02_fertilization'], 
                                             suffixes = ('_mean', '_std'))
    GAEZ_yb_stats = GAEZ_yb_stats.rename(columns = {
        'bin': 'Province', 
        'Value_mean': 'Yield_mean (t/ha)', 
        'Value_std': 'Yield_std (t/ha)'
        })
    
    GAEZ_yb_stats['Province'] = GAEZ_yb_stats['Province'].map(dict(enumerate(UNIQUE_VALUES['Province'])))
    
    
    plotnine.options.figure_size = (16, 6)
    plotnine.options.dpi = 100
    
    rcp = 'RCP2.6'
    co2 = 'With CO2 Fertilization'
    plot_df = GAEZ_yb_stats.query(f"rcp == '{rcp}' & c02_fertilization == '{co2}'").copy()
    plot_df['upper'] = plot_df['Yield_mean (t/ha)'] + plot_df['Yield_std (t/ha)'] * 1.96
    plot_df['lower'] = plot_df['Yield_mean (t/ha)'] - plot_df['Yield_std (t/ha)'] * 1.96
        
    g = (plotnine.ggplot(plot_df) +
         plotnine.geom_line(plotnine.aes(x = 'year', y = 'Yield_mean (t/ha)', color = 'water_supply')) +
         plotnine.geom_ribbon(plotnine.aes(x = 'year', ymin = 'lower', ymax = 'upper', fill = 'water_supply'), alpha = 0.2) +
         plotnine.facet_grid('crop~Province') +
         plotnine.theme_bw()
        )

