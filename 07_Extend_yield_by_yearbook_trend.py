import dask
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray as rxr
import plotnine
import dask
import dask.array as da

from helper_func.GAEZ_2_download import GAEZ
from dask.diagnostics import ProgressBar as Progressbar



# Read data
yearbook_trend = xr.open_dataset('data/results/step_6_yearbook_yield_extrapolated.nc')
yearbook_trend_ratio = yearbook_trend / yearbook_trend.sel(year=2020)['mean']

# Read the GAEZ data
GAEZ_yield_mean = xr.open_dataset('data/results/step_5_GAEZ_actual_yield_extended.nc', chunks='auto')['data']
GAEZ_yield_std = xr.open_dataset('data/results/step_3_GAEZ_AY_GYGA_std.nc', chunks='auto')['data'].sel(year=slice(2020, 2101))

def generate_samples(mean, std, size):
    return da.random.normal(mean, std, size)

GAEZ_MC = xr.apply_ufunc(generate_samples, 
                         GAEZ_yield_mean, 
                         GAEZ_yield_std, 
                         1000,
                         input_core_dims=[[], [], []],
                         output_core_dims=[['sample']],
                         vectorize=True,
                         dask='parallelized',
                         dask_gufunc_kwargs={'output_sizes': {'sample': 1000}},)

# Adjust the yield by the yearbook trend
GAEZ_MC_adj_yb = GAEZ_MC * yearbook_trend_ratio['mean']

# Get the uncertainty range from the yearbook trend
yearbook_range = xr.DataArray(
    da.random.normal(0, 1, (1000,) + yearbook_trend_ratio['std'].shape),
    dims=('sample',) + yearbook_trend_ratio['std'].dims,
    coords={'sample': np.arange(1000), **yearbook_trend_ratio['std'].coords}
) * yearbook_trend_ratio['std']

# Add the yearbook trend uncertainty range to the yield
GAEZ_MC_adj_yb = GAEZ_MC_adj_yb + yearbook_range

with dask.config.set({'array.slicing.split_large_chunks': True}), Progressbar():
    GAEZ_mean = GAEZ_MC_adj_yb.mean('sample').compute()
    GAEZ_std = GAEZ_MC_adj_yb.std('sample').compute()




# Sankity Check
if __name__ == '__main__':
    
    # Multiply the yield with mask mean to get the yield for each province
    GAEZ_province_yield_mean_prcsoy = np.einsum('prcsoyhw, phw -> prcsoy', GAEZ_yield_mean_prcsoyhw, mask_mean)     # (p, r, c, s, o, y)
    GAEZ_province_yield_std_prcsoy = np.einsum('prcsoyhw, phw -> prcsoy', GAEZ_yield_std_prcsoyhw, mask_mean)       # (p, r, c, s, o, y)
    
    # Convert nan to zero
    GAEZ_province_yield_std_prcsoy = np.nan_to_num(GAEZ_province_yield_std_prcsoy)
    
    # Convert the ndarray to a dataframe
    GAEZ_province_yield_mean_prcsoy_df = ndarray_to_df(GAEZ_province_yield_mean_prcsoy,'prcsoy')
    GAEZ_province_yield_std_prcsoy_df = ndarray_to_df(GAEZ_province_yield_std_prcsoy,'prcsoy')
    
    GAEZ_province_yield_df = pd.merge(GAEZ_province_yield_mean_prcsoy_df, 
                                      GAEZ_province_yield_std_prcsoy_df, 
                                      on=['Province', 'rcp', 'crop', 'water_supply', 'c02_fertilization', 'simulation_year'], 
                                      suffixes=('_mean', '_std'))
    
    GAEZ_province_yield_df['ci_low'] = GAEZ_province_yield_df['Value_mean'] - 1.96 * GAEZ_province_yield_df['Value_std']
    GAEZ_province_yield_df['ci_high'] = GAEZ_province_yield_df['Value_mean'] + 1.96 * GAEZ_province_yield_df['Value_std']
    
    
    
    # Read the yearbook yield historical data
    yearbook_yield_hist_df = pd.read_csv('data/results/yearbook_yield.csv')
    yearbook_yield_hist_df = yearbook_yield_hist_df.query('year >= 1990')
    
    
    # Plot the yield
    plotnine.options.figure_size = (16, 6)
    plotnine.options.dpi = 100
    
    rcp = 'RCP4.5'
    GAEZ_province_yield_df_45 = GAEZ_province_yield_df.query('rcp == @rcp')
    
    g = (plotnine.ggplot() +
        plotnine.geom_line(GAEZ_province_yield_df_45,
                           plotnine.aes(x='simulation_year', 
                                        y='Value_mean', 
                                        color='water_supply', 
                                        linetype='c02_fertilization')
                           ) +
        plotnine.geom_ribbon(GAEZ_province_yield_df_45,
                             plotnine.aes(x='simulation_year', 
                                          ymin='ci_low', 
                                          ymax='ci_high',
                                          fill='water_supply', 
                                          color='water_supply',
                                          linetype='c02_fertilization'), alpha=0.5
                             ) +
        plotnine.geom_point(yearbook_yield_hist_df,
                            plotnine.aes(x='year', y='Yield (tonnes)'),
                            size=0.2, alpha=0.3) +
        plotnine.facet_grid('crop~Province', scales='free_y') +
        plotnine.theme(axis_text_x=plotnine.element_text(rotation=90))
        )



