import numpy as np
import pandas as pd
from rasterio import band
import rioxarray
import xarray as xr
import rioxarray as rxr
import plotnine

from tqdm.auto import tqdm
from helper_func import compute_mean_std
from helper_func.calculate_GAEZ_stats import bincount_with_mask
from helper_func.parameters import GAEZ_variables, GAEZ_year_mid, UNIQUE_VALUES

# Get the group variables to group the GAEZ data
group_vars = GAEZ_variables['GAEZ_4'].copy()
group_vars = [i for i in group_vars if i not in ['model']]  # Remove the `model` variable


# Read the GAEZ_df which records the metadata of the GAEZ data
GAEZ_df = pd.read_csv('data/GAEZ_v4/GAEZ_df.csv').replace(GAEZ_year_mid)    # Replace the year with the mid year
GAEZ_df = GAEZ_df.query('GAEZ == "GAEZ_4" and year != "1981-2010"')         # Remove historical data



# Group by the group_vars and compute the mean and std of attainable yield
means = []
stds = []
for idx, df in tqdm(list(GAEZ_df.groupby(group_vars))):
    # Get tif paths
    tif_paths = df['fpath'].tolist()
    attrs = dict(zip(group_vars, [[i] for i in idx]))
    # Compute the mean and std for all tifs
    mean, std = compute_mean_std(tif_paths)
    means.append(mean.expand_dims(**attrs))
    stds.append(std.expand_dims(**attrs))    


# Combine the mean and std, multiply by the mask to convert the nodata to 0
mask_GAEZ = rioxarray.open_rasterio('data/GAEZ_v4/GAEZ_mask.tif')
mean_xr = xr.combine_by_coords(means) * mask_GAEZ  # kg/ha
std_xr = xr.combine_by_coords(stds) * mask_GAEZ

# Interpolate the mean and std to 5 year interval
mean_xr = mean_xr.interp(year=range(2010,2101,5), method='linear', kwargs={"fill_value": "extrapolate"})
std_xr = std_xr.interp(year=range(2010,2101,5), method='linear', kwargs={"fill_value": "extrapolate"})
std_xr = std_xr.where(std_xr > 0, 0)  # Set the negative std to 0

# Save with compression
mean_xr.name = 'data'
std_xr.name = 'data'
encoding = {'data': {"compression": "gzip", "compression_opts": 9}}

mean_xr.to_netcdf('data/results/step_1_GAEZ_4_attain_mean.nc', encoding=encoding, engine='h5netcdf')
std_xr.to_netcdf('data/results/step_1_GAEZ_4_attain_std.nc', encoding=encoding, engine='h5netcdf')



# Plot the attainable yield for each province
if __name__ == '__main__':
    # Get the mask
    mask_province = rioxarray.open_rasterio('data/GAEZ_v4/Province_mask.tif')
    mask_mean = rxr.open_rasterio('data/GAEZ_v4/Province_mask_mean.tif').sel(band=1).drop_vars('band')

    # bincount the mask to get the sum for each province
    mean_xr_t = mean_xr * mask_mean / 1e3                # t/ha

    # Convert to DataFrame
    bincount_df = bincount_with_mask(mask_province, mean_xr_t)
    bincount_df['bin'] = bincount_df['bin'].map(lambda x:UNIQUE_VALUES['Province'][int(x)])
    bincount_df = bincount_df.rename(columns={'Value':'Dry Weight yield (t/ha)', 'bin':'Province'})

    
    # Plot the attainable yield for each province
    plotnine.options.figure_size = (10, 6)
    plotnine.options.dpi = 100
    
    g = (plotnine.ggplot(bincount_df.query('rcp == "RCP2.6"')) +
         plotnine.geom_line(plotnine.aes(x='year', y='Dry Weight yield (t/ha)', color='Province',linetype='water_supply')) +
         plotnine.facet_grid('crop~c02_fertilization') +
         plotnine.theme_bw()
        )
    
    g.save('data/results/fig_step_1_GAEZ_attainaiable_yiled_dry_weight_t_ha.svg')


