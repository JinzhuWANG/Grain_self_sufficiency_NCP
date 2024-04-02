import numpy as np
import pandas as pd
import rasterio
import plotnine
import dask.array as da

from helper_func import ndarray_to_df, sample_from_mean_std




# Read the GAEZ_mask
with rasterio.open('data/GAEZ_v4/Province_mask.tif') as src:
    mask = src.read()                                                     # (p, h, w)

with rasterio.open('data/GAEZ_v4/Province_mask_mean.tif') as src:
    mask_mean = src.read()                                                # (p, h, w)



# Read the GAEZ_yield being propogated by GAEZ_attainable yield
GAEZ_yield_t_mean_prcsoyhw = np.load('data/results/GAEZ_yield_base_target_prcsoyhw.npy')           # (p, r, c, s, o, y, h, w)

# Read the GAEZ_std yield, multiply it with the mask, and remove the first two years to match the GAEZ_yield_mean
GAEZ_yield_t_std_rcsoyhw = np.load('data/results/GAEZ_4_attain_extrapolated_std_t_rcsoyhw.npy')         # (r, c, s, o, y, h, w)
GAEZ_yield_t_std_prcsoyhw = np.einsum('rcsoyhw, phw -> prcsoyhw', GAEZ_yield_t_std_rcsoyhw, mask)       # (p, r, c, s, o, y, h, w)
GAEZ_yield_t_std_prcsoyhw = GAEZ_yield_t_std_prcsoyhw[:, :, :, :, :, 2:, :, :]                          # (p, r, c, s, o, y, h, w)


# Read the yearbook yield projection data
yearbook_yield_mean_fitted_scale_pcsy = np.load('data/results/yearbook_yield_mean_fitted_scale_pcsy.npy')   # (p, c, s, y)
yearbook_yield_std_fitted_scale_pcsy = np.load('data/results/yearbook_yield_std_fitted_scale_pcsy.npy')     # (p, c, s, y)


# Convert the yearbook_yield mean and std to the same shape as the GAEZ TIF data
yearbook_yield_mean_fitted_scale_pcsyhw = np.einsum('pcsy, phw -> pcsyhw',
                                                yearbook_yield_mean_fitted_scale_pcsy,
                                                mask)                                                       # (p, c, s, y, h, w)

yearbook_yield_std_fitted_scale_pcsyhw = np.einsum('pcsy, phw -> pcsyhw',
                                                  yearbook_yield_std_fitted_scale_pcsy,
                                                  mask)                                                     # (p, c, s, y, h, w)



# Sample from the mean and std
yearbook_scale_sample_npcsyhw = sample_from_mean_std(yearbook_yield_mean_fitted_scale_pcsyhw, yearbook_yield_std_fitted_scale_pcsyhw)         # (n, p, c, s, y, h, w)
GAEZ_sample_nprcsoyhw = sample_from_mean_std(GAEZ_yield_t_mean_prcsoyhw, GAEZ_yield_t_std_prcsoyhw)                                     # (n, p, r, c, s, o, y, h, w)

# Multiply the yearbook_sample with the GAEZ_scale_sample
GAEZ_yield = da.einsum('nprcsoyhw, npcsyhw -> nprcsoyhw', GAEZ_sample_nprcsoyhw, yearbook_scale_sample_npcsyhw)                         # (n, p, r, c, s, o, y, h, w)

# Compute the mean and std of the GAEZ_yield, put them in the same computation graph to avoid recomputation
GAEZ_yield_mean_prcsoyhw, GAEZ_yield_std_prcsoyhw = da.compute(da.mean(GAEZ_yield, axis=0), da.std(GAEZ_yield, axis=0))     # (p, r, c, s, o, y, h, w)


# Save the GAEZ_yield to a file
np.save('data/results/GAEZ_yield_mean_prcsoyhw.npy', GAEZ_yield_mean_prcsoyhw.astype(np.float16))
np.save('data/results/GAEZ_yield_std_prcsoyhw.npy', GAEZ_yield_std_prcsoyhw.astype(np.float16))



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



