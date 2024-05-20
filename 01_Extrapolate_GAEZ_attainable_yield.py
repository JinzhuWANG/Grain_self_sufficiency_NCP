import numpy as np
import pandas as pd
import rioxarray
import xarray as xr

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
mean_xr = xr.combine_by_coords(means) * mask_GAEZ  # kg/ha
std_xr = xr.combine_by_coords(stds) * mask_GAEZ

# Interpolate the mean and std to 5 year interval
mean_xr = mean_xr.interp(year=range(2010,2101,5), method='linear', kwargs={"fill_value": "extrapolate"})
std_xr = std_xr.interp(year=range(2010,2101,5), method='linear', kwargs={"fill_value": "extrapolate"})


# Save with compression
mean_xr.name = 'data'
std_xr.name = 'data'
encoding = {'data': {"compression": "gzip", "compression_opts": 9}}

mean_xr.to_netcdf('data/results/step_1_GAEZ_4_attain_mean.nc', encoding=encoding, engine='h5netcdf')
std_xr.to_netcdf('data/results/step_1_GAEZ_4_attain_std.nc', encoding=encoding, engine='h5netcdf')




if __name__ == '__main__':
    # Get the mask
    mask_province = rioxarray.open_rasterio('data/GAEZ_v4/Province_mask.tif')
    GAEZ_area = rioxarray.open_rasterio('data/GAEZ_v4/GAEZ_area_km2.tif')       # km2

    # bincount the mask to get the sum for each province
    mean_xr_t = mean_xr * GAEZ_area / 10                # (kg/ha) * km2 / 10 = t
    
    # Define a function to perform bincount with weights
    def weighted_bincount(data, weights, minlength=None):
        return np.bincount(data.ravel(), weights=weights.ravel(), minlength=minlength)

    # Apply the function using xr.apply_ufunc
    bincount_result = xr.apply_ufunc(
        weighted_bincount,
        mask_province,
        mean_xr_t,
        input_core_dims=[['band', 'y', 'x'], ['band', 'y', 'x']],
        output_core_dims=[['bin']],
        vectorize=True,
        dask='allowed',
        output_dtypes=[float],
        kwargs={'minlength': int(mask_province.max().values) + 1}  # Ensure bins for all unique mask values
    )

    # Assign a name to the DataArray
    bincount_result.name = 'bincount'

    # Convert to DataFrame
    bincount_df = bincount_result.to_dataframe().reset_index()
    bincount_df['bin'] = bincount_df['bin'].map(lambda x:UNIQUE_VALUES['Province'][int(x)])

    import plotnine
    g = (plotnine.ggplot(bincount_df.query('rcp == "RCP2.6"')) +
         plotnine.geom_line(plotnine.aes(x='year', y='bincount',color='bin',linetype='water_supply')) +
         plotnine.facet_grid('crop~c02_fertilization') +
         plotnine.theme_bw()
        )

