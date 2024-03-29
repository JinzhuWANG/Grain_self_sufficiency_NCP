import numpy as np
import pandas as pd
import rasterio

from helper_func import ndarray_to_df




# Read the yearbook data
yearbook_area_pc = np.load('data/results/yearbook_area_targ_pc.npy')       # (p, c)
yearbook_yield_pc = np.load('data/results/yearbook_yield_targ_pc.npy')     # (p, c)

yearbook_area_pc_df = ndarray_to_df(yearbook_area_pc, 'pc')
yearbook_yield_pc_df = ndarray_to_df(yearbook_yield_pc, 'pc')


# Read the GAEZ actual yiled data in the base
GAEZ_area_pcshw = np.load('data/results/GAEZ_base_yr_area_pcshw.npy')         # (p, c, s, h, w)
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


out_pcs = np.einsum('pcshw, phw -> pcs', GAEZ_yield_pcshw, mask_mean)   
out_pcs_df = ndarray_to_df(out_pcs, 'pcs')





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
GAEZ_yield_base_target_rcsoyhw = np.einsum('pcshw, rcsoyhw -> rcsoyhw',
                                            GAEZ_yield_pcshw, 
                                            GAEZ_attain_yield_mean_multiplier_rcsoyhw)    # (r, c, s, o, y, h, w)




# Sanity Check
if __name__ == '__main__':
    
    # Get the avg yield for each province
    GAEZ_yield_base_target_rcsoyp = np.einsum('rcsoyhw, phw -> rcsoyp', 
                                             GAEZ_yield_base_target_rcsoyhw, 
                                             mask_mean)                                 # (r, c, s, o, y, p)
    
    GAEZ_yield_base_target_rcsoy_df = ndarray_to_df(GAEZ_yield_base_target_rcsoyp, 'rcsoyp')



