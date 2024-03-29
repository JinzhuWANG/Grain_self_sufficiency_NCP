import numpy as np
import pandas as pd
import plotnine
import rasterio

from helper_func import ndarray_to_df




# Read the yearbook data
yearbook_yield = pd.read_csv('data/results/yearbook_yield.csv')


# Read the GAEZ actual yiled data in the base
GAEZ_yield_pcshw = np.load('data/results/GAEZ_base_yr_yield_pcshw.npy')       # (p, c, s, h, w)
# Read the GAEZ_attainable_potential data
GAEZ_attain_yield_mean_all_yr_rcsoyhw = np.load('data/results/GAEZ_4_attain_extrapolated_mean_t_rcsoyhw.npy')       # (r, c, s, o, y, h, w)
# Remove the 2010, 2015 form the attainable yield
GAEZ_attain_yield_mean_rcsoyhw = GAEZ_attain_yield_mean_all_yr_rcsoyhw[:, :, :, :, 2:, :, :]    # (r, c, s, o, y, h, w)




# Read the mask of the GAEZ data
with rasterio.open('data/GAEZ_v4/Province_mask.tif') as src:
    mask = src.read()                                                # (p, h, w)

with rasterio.open('data/GAEZ_v4/Province_mask_mean.tif') as src:
    mask_mean = src.read()                                           # (p, h, w)




# Calculate the scale change of the attainable yield between the base and the future
GAEZ_attain_yield_mean_base_rcsohw = GAEZ_attain_yield_mean_rcsoyhw[:, :, :, :, 0, :, :]             # (r, c, s, o, h, w)
GAEZ_attain_yield_mean_base_divider_rcsohw = 1/GAEZ_attain_yield_mean_base_rcsohw                    # (r, c, s, o, h, w)

# Update the result with the mask to avoid the <division by zero>
GAEZ_attain_yield_mean_base_divider_rcsohw = np.einsum('rcsohw, phw -> rcsohw',
                                                       GAEZ_attain_yield_mean_base_divider_rcsohw, 
                                                       mask)                                         # (r, c, s, o, h, w)


# Divide the attainable yield by its base year value to get the multiplier
GAEZ_attain_yield_mean_multiplier_rcsoyhw = np.einsum('rcsoyhw, rcsohw -> rcsoyhw', 
                                                      GAEZ_attain_yield_mean_rcsoyhw, 
                                                      GAEZ_attain_yield_mean_base_divider_rcsohw)    # (r, c, s, o, y, h, w)



# We assume the yield will NOT decrease in the future, so we manually set all values >= 1
GAEZ_attain_yield_mean_multiplier_rcsoyhw[GAEZ_attain_yield_mean_multiplier_rcsoyhw < 1] = 1

# Replace the NaN values with 0
GAEZ_attain_yield_mean_multiplier_rcsoyhw = np.nan_to_num(GAEZ_attain_yield_mean_multiplier_rcsoyhw)



# Apply the multiplier to the GAEZ actual yield
GAEZ_yield_base_target_prcsoyhw = np.einsum('pcshw, rcsoyhw -> prcsoyhw',
                                            GAEZ_yield_pcshw, 
                                            GAEZ_attain_yield_mean_multiplier_rcsoyhw)    # (p, r, c, s, o, y, h, w)




# Sanity Check
if __name__ == '__main__':
    
    # Get the avg yield for each province
    GAEZ_yield_base_target_rcsoyp = np.einsum('prcsoyhw, phw -> rcsoyp', 
                                             GAEZ_yield_base_target_prcsoyhw, 
                                             mask_mean)                                 # (r, c, s, o, y, p)
    
    GAEZ_yield_base_target_rcsoy_df = ndarray_to_df(GAEZ_yield_base_target_rcsoyp, 'rcsoyp')
    
    
    # Make a plot
    plotnine.options.figure_size = (12, 8)
    plotnine.options.dpi = 100
    
    rcp = 'RCP4.5'
    GAEZ_yield_base_target_rcsoy_df_rcp45 = GAEZ_yield_base_target_rcsoy_df.query('rcp == @rcp')
    
    g = (plotnine.ggplot() +
         plotnine.geom_line(GAEZ_yield_base_target_rcsoy_df_rcp45, 
                            plotnine.aes(x='simulation_year', 
                                         y='Value', 
                                         color='water_supply', 
                                         linetype='c02_fertilization') ) + 
         plotnine.geom_point(yearbook_yield, plotnine.aes(x='year', y='Yield (tonnes)')) +
         plotnine.facet_grid('crop~Province', scales='free_y') +
        #  plotnine.theme(legend_position='none') +
         plotnine.ggtitle('GAEZ Yield Base Target')
         )
    


