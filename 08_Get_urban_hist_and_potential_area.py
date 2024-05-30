import numpy as np
import pandas as pd
import xarray as xr
import rioxarray as rxr
import dask
import dask.array as da
import plotnine

from helper_func.calculate_GAEZ_stats import weighted_bincount
from helper_func.parameters import UNIQUE_VALUES, BLOCK_SIZE
from dask.diagnostics import ProgressBar

# Define the working chunk size
work_size = BLOCK_SIZE * 8


# Read data
mask_province = xr.open_dataset('data/LUCC/LUCC_Province_mask.nc', chunks={'y': work_size, 'x': work_size})['data']
mask = xr.where(mask_province >= 0, 1, 0)

urban_arr = xr.open_dataset('data/LUCC/Norm_Urban_1990_2019.nc', chunks={'y': work_size, 'x': work_size})['data']



# Encode year and province to a single array
year_base = 100

urban_year_region_max =  mask_province.max().compute() * year_base + urban_arr.max().compute()
urban_potential_region_max = mask_province.max().compute() * potential_base + urban_potential_arr.max().compute()





area_hist = da.map_blocks(
    weighted_bincount, 
    urban_year_region, 
    lucc_area_area,
     
    dtype=np.float32)




with ProgressBar():
    # Compute the bincount for the province and append it to the list
    area_hist = da.bincount(urban_year_region.ravel(), weights=lucc_area_area.ravel())
    area_potential = da.bincount(urban_potential_region.ravel(), weights=lucc_area_area.ravel())
    area_hist, area_potential = dask.compute(area_hist, area_potential)
    



# Get the historical urban area
area_hist_df = pd.DataFrame(enumerate(area_hist), columns=['encode', 'Area_km2'])
area_hist_df['Province_code'] = area_hist_df['encode'] // year_base
area_hist_df['year_code'] = area_hist_df['encode'] % year_base

area_hist_df['Province'] = area_hist_df['Province_code'].map(dict(enumerate(UNIQUE_VALUES['Province'])))
area_hist_df['year'] = area_hist_df['year_code'].map({k+1:v for k, v in enumerate(UNIQUE_VALUES['Urban_map_year'][::-1])})

area_hist_df = area_hist_df.dropna()
area_hist_df['year'] = area_hist_df['year'].astype(int)

area_hist_df = area_hist_df.sort_values(['Province', 'year'])
area_hist_df['Area_cumsum_km2'] = area_hist_df.groupby('Province')['Area_km2'].cumsum()
area_hist_df = area_hist_df[['Province', 'year', 'Area_cumsum_km2']]


# Get the potential urban area
urban_area_potential = pd.DataFrame(enumerate(area_potential), columns=['encode', 'Area_km2'])
urban_area_potential['Province_code'] = urban_area_potential['encode'] // potential_base
urban_area_potential['Potential'] = urban_area_potential['encode'] % potential_base

urban_area_potential['Province'] = urban_area_potential['Province_code'].map(dict(enumerate(UNIQUE_VALUES['Province'])))
urban_area_potential = urban_area_potential.sort_values(['Province', 'Potential'], ascending=[True, False])

urban_area_potential = urban_area_potential.dropna().reset_index(drop=True)
urban_area_potential['Area_cumsum_km2'] = urban_area_potential.groupby('Province')['Area_km2'].cumsum()
urban_area_potential = urban_area_potential[['Province', 'Potential', 'Area_cumsum_km2']]



# Convert the list to a Dask array
area_hist_df.to_csv('data/results/area_hist_df.csv', index=False)
urban_area_potential.to_csv('data/results/urban_area_potential.csv', index=False)

# Read the data
urban_area_hist = pd.read_csv('data/results/urban_area_hist.csv')
urban_area_potential = pd.read_csv('data/results/urban_area_potential.csv')



# Sanity check
if __name__ == '__main__':
    plotnine.options.figure_size = (12, 8)
    plotnine.options.dpi = 100
    
    g = (plotnine.ggplot(area_hist_df) +
        plotnine.aes(x='year', y='Area_cumsum_km2', color='Province') +
        plotnine.geom_line() +
        plotnine.theme_bw() +
        plotnine.theme(axis_text_x=plotnine.element_text(rotation=45, hjust=1))
        )
    
    g = (plotnine.ggplot(urban_area_potential.query('Potential > 9900')) +
        plotnine.aes(x='Potential', y='Area_cumsum_km2', color='Province') +
        plotnine.geom_line() +
        plotnine.theme_bw() +
        plotnine.theme(axis_text_x=plotnine.element_text(rotation=45, hjust=1)) +
        plotnine.scale_x_reverse()
        )


