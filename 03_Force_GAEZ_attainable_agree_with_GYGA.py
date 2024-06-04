import math
import pandas as pd
import plotnine
import rioxarray as rxr
import xarray as xr

from helper_func.calculate_GAEZ_stats import bincount_with_mask
from helper_func.get_yearbook_records import get_yearbook_yield
from helper_func.parameters import UNIQUE_VALUES


# Read stats
GYGA_PY = pd.read_csv('data/results/step_3_GYGA_attainable.csv')        # t/ha
GYGA_PY['year'] = 2010
GYGA_PY_2010 = GYGA_PY.groupby(['Province','crop','water_supply']).mean(numeric_only=True).reset_index()

GAEZ_attain = pd.read_csv('data/results/step_2_GAEZ_attainable.csv')    # t/ha
GAEZ_attain = GAEZ_attain.rename(columns={'bin' : 'Province'})

GYGA_GAEZ_2010 = GYGA_PY_2010.merge(GAEZ_attain, on=['Province', 'crop', 'water_supply','year'], how='left')
GYGA_GAEZ_2010['GAEZ2GYGA_mul'] = GYGA_GAEZ_2010['yield_potential_adj'] / GYGA_GAEZ_2010['Yield t/ha_mean']
GYGA_GAEZ_2010 = GYGA_GAEZ_2010.set_index(['Province', 'crop', 'water_supply', 'rcp', 'c02_fertilization', ])[['GAEZ2GYGA_mul']]

GAEZ_mul = xr.Dataset.from_dataframe(GYGA_GAEZ_2010)


# Read the GAEZ_attainable_yield_t data
mask_province = rxr.open_rasterio('data/GAEZ_v4/Province_mask.tif')
mask_province = [(mask_province == idx).expand_dims({'Province': [p]}) 
                 for idx,p in enumerate(UNIQUE_VALUES['Province'])]
mask_province = xr.combine_by_coords(mask_province)
mask_province = mask_province * GAEZ_mul                                    # ratio/pixel

# Multiply the GAEZ_mean with the GAEZ2GYGA_mul
GAEZ_mean = xr.open_dataarray('data/results/step_2_GAEZ_4_attain_mean.nc')  # kg/ha
GAEZ_std = xr.open_dataarray('data/results/step_2_GAEZ_4_attain_std.nc')    # kg/ha

GAEZ_mean = GAEZ_mean * mask_province
GAEZ_std = GAEZ_std * mask_province

# Merge all province to a single map
GAEZ_mean = GAEZ_mean.sum(dim=['Province'])['GAEZ2GYGA_mul']
GAEZ_std = GAEZ_std.sum(dim=['Province'])['GAEZ2GYGA_mul']

# Save the data
GAEZ_mean.name = 'data'
GAEZ_std.name = 'data'
encoding = {'data': {"compression": "gzip", "compression_opts": 9}}
GAEZ_mean.to_netcdf('data/results/step_3_GAEZ_AY_GYGA_mean.nc', encoding=encoding, engine='h5netcdf')
GAEZ_std.to_netcdf('data/results/step_3_GAEZ_AY_GYGA_std.nc', encoding=encoding, engine='h5netcdf')



# Sanity check
if __name__ == '__main__':   
    # Read yield book
    yearbook_yield = get_yearbook_yield()  
    GAEZ_mask = rxr.open_rasterio('data/GAEZ_v4/GAEZ_mask.tif')
    mask_sum = rxr.open_rasterio('data/GAEZ_v4/Province_mask.tif')
    mask_mean = rxr.open_rasterio('data/GAEZ_v4/Province_mask_mean.tif')

    # Yield need to multiply with the mask_mean frist
    GAEZ_mean = GAEZ_mean * mask_mean
    GAEZ_std = GAEZ_std * mask_mean  

    # Apply the function using xr.apply_ufunc
    GAEZ_mean_df = bincount_with_mask(mask_sum, GAEZ_mean)
    GAEZ_std_df = bincount_with_mask(mask_sum, GAEZ_std)

    GAEZ_df = GAEZ_mean_df.merge(
        GAEZ_std_df, 
        on=['year', 'crop', 'water_supply', 'rcp', 'c02_fertilization', 'bin'],
        suffixes=('_mean', '_std')
        )
    
    GAEZ_df['Province'] = GAEZ_df['bin'].map(dict(enumerate(UNIQUE_VALUES['Province'])))
    GAEZ_df['Value_mean'] = GAEZ_df['Value_mean'] / 1e3     # kg/ha -> t/ha
    GAEZ_df['Value_std'] = GAEZ_df['Value_std'] / 1e3       # kg/ha -> t/ha

    GAEZ_df.to_csv('data/results/step_3_GAEZ_AY_GYGA_forcing.csv', index=False)   


    # Filter the yield_array with specific rcp
    rcp = 'RCP2.6'
    c02_fertilization = 'With CO2 Fertilization'
    GAEZ_PY_df = GAEZ_df.query(f"rcp == '{rcp}' & c02_fertilization == '{c02_fertilization}'").copy()
    GAEZ_PY_df['obs_ci_lower'] = GAEZ_PY_df['Value_mean'] - (GAEZ_PY_df['Value_std'] / math.sqrt(len(GAEZ_PY_df)) * 1.96)
    GAEZ_PY_df['obs_ci_upper'] = GAEZ_PY_df['Value_mean'] + (GAEZ_PY_df['Value_std'] / math.sqrt(len(GAEZ_PY_df)) * 1.96)
      
    # Plot the yield for each province of both yearbook and GAEZ
    plotnine.options.figure_size = (16, 6)
    plotnine.options.dpi = 100
    g = (
        plotnine.ggplot() +
        plotnine.geom_line(
            GAEZ_PY_df,
            plotnine.aes(
                x='year', 
                y='Value_mean', 
                color='water_supply',
            )
        ) +
        plotnine.geom_ribbon(
            GAEZ_PY_df,
            plotnine.aes(
                x='year', 
                ymin='obs_ci_lower', 
                ymax='obs_ci_upper', 
                fill='water_supply',
            ), 
            alpha=0.5
        ) +
        plotnine.geom_point(
            yearbook_yield, 
            plotnine.aes(
                x='year', 
                y='Yield (tonnes)'
                ),
            size=0.2,
            alpha=0.3) +
        plotnine.facet_grid('crop~Province', scales='free_y') +
        plotnine.theme(axis_text_x=plotnine.element_text(rotation=45, hjust=1))
    )
    
