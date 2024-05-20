import numpy as np
import pandas as pd
import rioxarray as rxr
import xarray as xr
import rasterio
import plotnine

from helper_func import ndarray_to_df
from helper_func.get_yearbook_records import get_yearbook_yield
from helper_func.parameters import UNIQUE_VALUES, Attainable_conversion
                                    

# Get the convertion factor for each crop <dry weight -> kg harvested>
convesion_factor = xr.DataArray(list(Attainable_conversion.values()), 
                                 dims=['crop'], 
                                 coords={'crop': list(Attainable_conversion.keys())})


# Read the GAEZ_extrapolated_df, multiply by the conversion factor (kg Dry weight -> kg harvested)
GAEZ_attain_mean = xr.open_dataarray('data/results/step_1_GAEZ_4_attain_mean.nc') * convesion_factor    # kg/ha
GAEZ_attain_std = xr.open_dataarray('data/results/step_1_GAEZ_4_attain_std.nc') *  convesion_factor     # kg/ha


# Save to disk
GAEZ_attain_mean.name = 'data'
GAEZ_attain_std.name = 'data'
encoding = {'data': {"compression": "gzip", "compression_opts": 9}}

GAEZ_attain_mean.to_netcdf('data/results/step_2_GAEZ_4_attain_mean.nc', encoding={'data': {"compression": "gzip", "compression_opts": 9}}, engine='h5netcdf')
GAEZ_attain_std.to_netcdf('data/results/step_2_GAEZ_4_attain_std.nc', encoding={'data': {"compression": "gzip", "compression_opts": 9}}, engine='h5netcdf')




# Compare the yield with the yearbook data
if __name__ == '__main__':
    
    # Read the yearbook_yield
    yearbook_yield = get_yearbook_yield()
    
    # Get mask_province
    mask_mean = rxr.open_rasterio('data/GAEZ_v4/Province_mask_mean.tif')
    mask_sum = rxr.open_rasterio('data/GAEZ_v4/Province_mask.tif')

    # Compute the mean yield
    mean_yield = GAEZ_attain_mean * mask_mean
    std_yield = GAEZ_attain_std * mask_mean


    # Define a function to perform bincount with weights
    def weighted_bincount(mask, weights, minlength=None):
        return np.bincount(mask.ravel(), weights=weights.ravel(), minlength=minlength)

    # Apply the function using xr.apply_ufunc
    mean_yield = xr.apply_ufunc(
        weighted_bincount,
        mask_sum,
        mean_yield,
        input_core_dims=[['band', 'y', 'x'], ['band', 'y', 'x']],
        output_core_dims=[['bin']],
        vectorize=True,
        dask='allowed',
        output_dtypes=[float],
        kwargs={'minlength': int(mask_sum.max().values) + 1}  # Ensure bins for all unique mask values
    )

    std_yield = xr.apply_ufunc(
        weighted_bincount,
        mask_sum,
        std_yield,
        input_core_dims=[['band', 'y', 'x'], ['band', 'y', 'x']],
        output_core_dims=[['bin']],
        vectorize=True,
        dask='allowed',
        output_dtypes=[float],
        kwargs={'minlength': int(mask_sum.max().values) + 1}  # Ensure bins for all unique mask values
    )


    # Assign a name to the DataArray
    mean_yield.name = 'bincount'
    std_yield.name = 'bincount'

    # Convert to DataFrame
    mean_df = mean_yield.to_dataframe().reset_index()
    mean_df['bin'] = mean_df['bin'].map(lambda x:UNIQUE_VALUES['Province'][int(x)])
    mean_df['bincount_t/ha'] = mean_df['bincount'] / 1e3             # kg/ha -> t/ha

    std_df = std_yield.to_dataframe().reset_index()
    std_df['bin'] = std_df['bin'].map(lambda x:UNIQUE_VALUES['Province'][int(x)])
    std_df['bincount_t/ha'] = std_df['bincount'] / 1e3               # kg/ha -> t/ha


    # Filter the yield_array with specific rcp, c02_fertilization, and water_supply
    rcp = "RCP4.5" 
    c02_fertilization = 'With CO2 Fertilization'

    GAEZ_yield_df = pd.merge(
        mean_df, 
        std_df, 
        on=['rcp', 'crop', 'year', 'water_supply', 'c02_fertilization', 'bin'], 
        suffixes=('_mean', '_std'))
    
    GAEZ_yield_df = GAEZ_yield_df.query(f"rcp == '{rcp}' and c02_fertilization == '{c02_fertilization}'")
    GAEZ_yield_df['obs_ci_lower'] = GAEZ_yield_df['bincount_t/ha_mean'] - (GAEZ_yield_df['bincount_t/ha_std'] * 1.96)
    GAEZ_yield_df['obs_ci_upper'] = GAEZ_yield_df['bincount_t/ha_mean'] + (GAEZ_yield_df['bincount_t/ha_std'] * 1.96)


    # Plot the yield for each province of both yearbook and GAEZ
    plotnine.options.figure_size = (16, 6)
    plotnine.options.dpi = 100
    g = (
        plotnine.ggplot() +
        plotnine.geom_line(GAEZ_yield_df,
                        plotnine.aes(x='year', y='bincount_t/ha_mean', color='water_supply')
                        ) +
        plotnine.geom_ribbon(GAEZ_yield_df,
                            plotnine.aes(x='year', ymin='obs_ci_lower', ymax='obs_ci_upper', fill='water_supply'), alpha=0.5
                            ) +
        plotnine.geom_point(yearbook_yield, plotnine.aes(x='year', y='Yield (tonnes)'), size=0.02, alpha=0.3) +
        plotnine.facet_grid('crop~bin', scales='free_y') +
        plotnine.theme(axis_text_x=plotnine.element_text(rotation=45, hjust=1))
    )


