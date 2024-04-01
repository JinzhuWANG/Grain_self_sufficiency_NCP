import numpy as np
import pandas as pd
import rasterio
import dask.array as da

from helper_func import sample_from_mean_std




# Read the GAEZ_mask
with rasterio.open('data/GAEZ_v4/Province_mask.tif') as src:
    mask = src.read()                                                     # (p, h, w)

with rasterio.open('data/GAEZ_v4/Province_mask_mean.tif') as src:
    mask_mean = src.read()                                                # (p, h, w)



# Read the GAEZ_yield in the BASE_YR
GAEZ_yield_t_mean_prcsoyhw = np.load('data/results/GAEZ_yield_base_target_prcsoyhw.npy')                # (p, r, c, s, o, y, h, w)

# Multiply the mask with the mean yield, and remove the first two years
GAEZ_yield_t_std_rcsoyhw = np.load('data/results/GAEZ_4_attain_extrapolated_std_t_rcsoyhw.npy')         # (r, c, s, o, y, h, w)
GAEZ_yield_t_std_prcsoyhw = np.einsum('rcsoyhw, phw -> prcsoyhw', GAEZ_yield_t_std_rcsoyhw, mask)       # (p, r, c, s, o, y, h, w)
GAEZ_yield_t_std_prcsoyhw = GAEZ_yield_t_std_prcsoyhw[:, :, :, :, :, 2:, :, :]                          # (p, r, c, s, o, y, h, w)


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




# Sample from the mean and std
yearbook_sample_npcyhw = sample_from_mean_std(yearbook_yield_mean_fitted_pcyhw, yearbook_yield_std_fitted_pcyhw, 2) # (n, p, c, y, h, w)
GAEZ_sample_nprcsoyhw = sample_from_mean_std(GAEZ_yield_t_mean_prcsoyhw, GAEZ_yield_t_std_prcsoyhw, 10)              # (n, p, r, c, s, o, y, h, w)


# Expand the yearbook_sample to the same shape as GAEZ_sample
yearbook_sample_nprcsoyhw = yearbook_sample_npcyhw[:, :, None, :, None, None, :, :, :] # (n, p, r, c, s, o, y, h, w)


# Add the yearbook_sample to the GAEZ_sample
GAEZ_yield = GAEZ_sample_nprcsoyhw + yearbook_sample_nprcsoyhw

# Compute the mean and std of the GAEZ_yield
GAEZ_yield_mean_prcsoyhw = da.mean(GAEZ_yield, axis=0).compute() # (p, r, c, s, o, y, h, w)
GAEZ_yield_std_prcsoyhw = da.std(GAEZ_yield, axis=0).compute()   # (p, r, c, s, o, y, h, w)


# Save the GAEZ_yield to a file
np.save('data/results/GAEZ_yield_mean_prcsoyhw.npy', GAEZ_yield_mean_prcsoyhw)
np.save('data/results/GAEZ_yield_std_prcsoyhw.npy', GAEZ_yield_std_prcsoyhw)








