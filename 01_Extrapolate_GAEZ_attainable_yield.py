import numpy as np
import pandas as pd
import rioxarray
import xarray as xr
from tqdm.auto import tqdm

from helper_func import compute_mean_std
from helper_func.parameters import GAEZ_variables, GAEZ_year_mid, UNIQUE_VALUES

# Compute the mean and std of attainable yield according to different climate models
group_vars = GAEZ_variables['GAEZ_4'].copy()
group_vars.remove('model')


# Read the GAEZ_df which records the metadata of the GAEZ data
GAEZ_df = pd.read_csv('data/GAEZ_v4/GAEZ_df.csv').replace(GAEZ_year_mid)    # Replace the year with the mid year
GAEZ_4_df = GAEZ_df.query('GAEZ == "GAEZ_4" and year != "1981-2010"')       # Remove historical data



# Group by the group_vars and compute the mean and std of attainable yield
means = []
stds = []
for idx, df in list(GAEZ_4_df.groupby(group_vars)):
    # Get tif paths
    tif_paths = df['fpath'].tolist()
    attrs = dict(zip(group_vars, [[i] for i in idx]))
    # Compute the mean and std for all tifs
    mean, std = compute_mean_std(tif_paths)
    means.append(mean.expand_dims(**attrs))
    stds.append(std.expand_dims(**attrs))    

# Combine the mean and std, multiply by the mask to convert the nodata to 0
mask_GAEZ = rioxarray.open_rasterio('data/GAEZ_v4/GAEZ_mask.tif')
mean_xr = xr.combine_by_coords(means) * mask_GAEZ
std_xr = xr.combine_by_coords(stds) * mask_GAEZ

# Interpolate the mean and std to 5 year interval
mean_xr = mean_xr.interp(year=range(2010,2101,5), method='linear', kwargs={"fill_value": "extrapolate"})
std_xr = std_xr.interp(year=range(2010,2101,5), method='linear', kwargs={"fill_value": "extrapolate"})



if __name__ == '__main__':
    # Get the mask
    mask_province = rioxarray.open_rasterio('data/GAEZ_v4/Province_mask.tif')

    # bincount the mask to get the sum for each province
    
