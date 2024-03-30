import numpy as np
import pandas as pd
import rasterio

from helper_func.parameters import Monte_Carlo_num




# Read the GAEZ_mask
with rasterio.open('data/GAEZ_v4/Province_mask.tif') as src:
    mask = src.read()                                                     # (p, h, w)

with rasterio.open('data/GAEZ_v4/Province_mask_mean.tif') as src:
    mask_mean = src.read()                                                # (p, h, w)



# Read the GAEZ_yield in the BASE_YR
GAEZ_yield_t_mean_prcsoyhw = np.load('data/results/GAEZ_yield_base_target_prcsoyhw.npy')                # (p, r, c, s, o, y, h, w)
GAEZ_yield_t_std_rcsoyhw = np.load('data/results/GAEZ_4_attain_extrapolated_std_t_rcsoyhw.npy')         # (r, c, s, o, y, h, w)
GAEZ_yield_t_std_prcsoyhw = np.einsum('rcsoyhw, phw -> prcsoyhw', GAEZ_yield_t_std_rcsoyhw, mask)       # (p, r, c, s, o, y, h, w)


# Read the yearbook yield projection data
yearbook_yield_mean_fitted_pcy = np.load('data/results/yearbook_yield_mean_fitted_pcy.npy')
yearbook_yield_std_fitted_pcy = np.load('data/results/yearbook_yield_std_fitted_pcy.npy')



# Convert the yearbook_yield mean and std to the same shape as the GAEZ TIF data
yearbook_yield_mean_fitted_pcyhw = np.einsum('pcy, phw -> pcyhw',
                                                yearbook_yield_mean_fitted_pcy,
                                                mask)                               # (p, c, y, h, w)

yearbook_yield_std_fitted_pcyhw = np.einsum('pcy, phw -> pcyhw',
                                                  yearbook_yield_std_fitted_pcy,
                                                  mask)                             # (p, c, y, h, w)



def sample_from_mean_std(mean:np.ndarray, std:np.ndarray, size: int = Monte_Carlo_num):
    
    if not mean.shape == std.shape:
        raise ValueError("The mean and std arrays must have the same shape")

    # Add an extra dimension to mean and std arrays
    mean_expanded = np.expand_dims(mean, 0).astype(np.float16)
    std_expanded = np.expand_dims(std, 0).astype(np.float16)

    # Generate samples
    sample_arr = np.random.normal(mean_expanded, std_expanded, (size,) + mean.shape).astype(np.float16)
    
    return sample_arr

out_1 = sample_from_mean_std(yearbook_yield_mean_fitted_pcyhw, yearbook_yield_std_fitted_pcyhw, 100)


















