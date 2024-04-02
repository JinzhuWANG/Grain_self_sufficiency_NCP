import numpy as np
import pandas as pd
import plotnine
import rasterio
from helper_func import ndarray_to_df
from helper_func.parameters import UNIQUE_VALUES
import matplotlib.pyplot as plt


# Define the columns for ordering the data
sort_volumns_GYGA = ['Province', 'crop','water_supply',]                    # p, c, s
unique_val_GYGA = [len(UNIQUE_VALUES[i]) for i in sort_volumns_GYGA]


# Read the GYGA attainable yield data
GYGA_PY = pd.read_csv('data/GYGA/GYGA_attainable_filled.csv').replace('Rainfed', 'Dryland')
GYGA_PY_2010 = GYGA_PY.groupby(['Province','crop','water_supply']).mean(numeric_only=True).reset_index()
GYGA_PY_2010 = GYGA_PY_2010.sort_values(by=sort_volumns_GYGA).reset_index(drop=True)
GYGA_PY_2010_pcs = GYGA_PY_2010['yield_potential'].values.reshape(*unique_val_GYGA)                     # (p, c, s) province, crop, water_supply






# Read the GAEZ attainable yield data
GAEZ_PY_mean_t_rcsoyhw = np.load('data/results/GAEZ_4_attain_extrapolated_mean_t_rcsoyhw.npy')          # (r, c, s, o, y, h, w)
GAEZ_PY_2010_rcsohw = GAEZ_PY_mean_t_rcsoyhw[:, :, :, :, UNIQUE_VALUES['attainable_year'].index(2010), :, :]       # (r, c, s, o, h, w)

# Read the mean mask
with rasterio.open('data/GAEZ_v4/Province_mask_mean.tif') as src:
    mask_mean = src.read()                                                                              # (p, h, w) 
    
with rasterio.open('data/GAEZ_v4/Province_mask.tif') as src:
    mask = src.read()                                                                                   # (p, h, w)

# Compute the mean GAEZ attainable yield for 2010
GAEZ_PY_2010_mean_rcsop = np.einsum('rcsohw,phw->rcsop', GAEZ_PY_2010_rcsohw, mask_mean)                # (r, c, s, o, p) 






# Rearrange the dimensions of GAEZ_PY_2010_mean_rcsop to match GYGA_PY_2010_pcs
GAEZ_PY_2010_mean_pcsro = np.transpose(GAEZ_PY_2010_mean_rcsop, (4, 1, 2, 0, 3))                        # (p, c, s, r, o)

# Now you can subtract GAEZ_PY_2010_mean from GYGA_PY_2010_pcs for each r and o
diff_PY_2010_pcsro = GYGA_PY_2010_pcs[:, :, :, None, None] - GAEZ_PY_2010_mean_pcsro                    # (p, c, s, r, o)

# Spred the diff_PY_2010 to the mask_mean
diff_PY_2010_rcsohw = np.einsum('pcsro,phw->rcsohw', diff_PY_2010_pcsro, mask)                          # (r, c, s, o, h, w)

# Add the difference to all GAEZ_attainable_yield_t
GAEZ_PY_mean_t_GYGA_rcsoyhw = GAEZ_PY_mean_t_rcsoyhw + diff_PY_2010_rcsohw[:,:,:,:,None,:,:,]           # (r, c, s, o, y, h, w)



# Save the GAEZ_PY_mean_t_GYGA_rcsoyhw
np.save('data/results/GAEZ_PY_mean_t_GYGA_rcsoyhw.npy', GAEZ_PY_mean_t_GYGA_rcsoyhw.astype(np.float16))





# Sanity check
if __name__ == '__main__':
    
    
    
    # Read yield book
    yearbook_yield = pd.read_csv('data/results/yearbook_yield.csv')
    
    
    # Convert the GAEZ_PY_mean_t_GYGA_rcsoyhw to a dataframe
    GAEZ_PY_mean_t_GYGA_rcsoyp = np.einsum('rcsoyhw,phw->rcsoyp', GAEZ_PY_mean_t_GYGA_rcsoyhw, mask_mean)   # (r, c, s, o, y, p)
    
    # Get the std of GAEZ_PY_mean_t_GYGA_rcsoyhw
    GAEZ_PY_std_t_GYGA_rcsoyhw = np.load('data/results/GAEZ_4_attain_extrapolated_std_t_rcsoyhw.npy')       # (r, c, s, o, y, h, w)
    GAEZ_PY_std_t_GYGA_rcsoyp = np.einsum('rcsoyhw,phw->rcsoyp', GAEZ_PY_std_t_GYGA_rcsoyhw, mask_mean)     # (r, c, s, o, p, y)
    
    GAEZ_PY_mean_t_GYGA_rcsoyp_df = ndarray_to_df(GAEZ_PY_mean_t_GYGA_rcsoyp, 'rcsoyp', year_start=2010)
    GAEZ_PY_std_t_rcsoyp_df = ndarray_to_df(GAEZ_PY_std_t_GYGA_rcsoyp, 'rcsoyp', year_start=2010)
    
    GAEZ_PY_df = pd.merge(GAEZ_PY_mean_t_GYGA_rcsoyp_df, 
                          GAEZ_PY_std_t_rcsoyp_df, 
                          on=['rcp', 'crop', 'water_supply', 'c02_fertilization', 'attainable_year','Province'],
                          suffixes=('_mean', '_std'))

    # Filter the yield_array with specific rcp
    rcp = 'RCP4.5'
    GAEZ_PY_df = GAEZ_PY_df.query(f"rcp == '{rcp}'")
    GAEZ_PY_df['obs_ci_lower'] = GAEZ_PY_df['Value_mean'] - (GAEZ_PY_df['Value_std'] * 1.96)
    GAEZ_PY_df['obs_ci_upper'] = GAEZ_PY_df['Value_mean'] + (GAEZ_PY_df['Value_std'] * 1.96)
      
    # Plot the yield for each province of both yearbook and GAEZ
    plotnine.options.figure_size = (16, 6)
    plotnine.options.dpi = 100
    g = (
        plotnine.ggplot() +
        plotnine.geom_line(GAEZ_PY_df,
                        plotnine.aes(x='attainable_year', 
                                     y='Value_mean', 
                                     color='water_supply', 
                                     linetype='c02_fertilization' )
                        ) +
        plotnine.geom_ribbon(GAEZ_PY_df,
                        plotnine.aes(x='attainable_year', 
                                     ymin='obs_ci_lower', 
                                     ymax='obs_ci_upper', 
                                     linetype='c02_fertilization', 
                                     fill='water_supply'),
                        alpha=0.5
                        ) +
        plotnine.geom_point(yearbook_yield, plotnine.aes(x='year', y='Yield (tonnes)')) +
        plotnine.facet_grid('crop~Province', scales='free_y') +
        plotnine.theme(axis_text_x=plotnine.element_text(rotation=45, hjust=1))
    )
    
