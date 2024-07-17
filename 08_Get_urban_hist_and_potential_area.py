import pandas as pd
import xarray as xr
import dask
import dask.array as da
import plotnine

from helper_func.parameters import UNIQUE_VALUES, BLOCK_SIZE
from dask.diagnostics import ProgressBar

# Define the working chunk size
work_size = BLOCK_SIZE * 8


# Read data
mask_province = xr.open_dataset('data/LUCC/LUCC_Province_mask.nc', chunks={'y': work_size, 'x': work_size})['data']
mask_province = xr.where(mask_province<0, 0, mask_province).astype(int)

urban_arr = xr.open_dataset('data/LUCC/Norm_Urban_1990_2019.nc', chunks={'y': work_size, 'x': work_size})['data'].astype('float32')
urban_potential = xr.open_dataset('data/LUCC/Norm_Transition_potential.nc', chunks={'y': work_size, 'x': work_size})['data'].astype('float32')
urban_area = xr.open_dataset('data/LUCC/LUCC_area_km2.nc', chunks={'y': work_size, 'x': work_size})['data'].astype('float32')


# Encode year and province to a single array
'''
Here we use a multiplication method to encode the year and province to a single array.
    - The historical urban is an 2D array ranged from 0 to 10, indicating year of 2018, 2015, ..., 1991.
    - The mask_province is a 2D array ranged from 0 to 6, indicating the province code.
    
By `mask_province` * `year_mul` + `urban_arr`, we can encode the year and province to a single array.
For example, if a resulted cell has a value of 305, we then now that it must comes form `mask_province` (3) and and historical (5).
'''
year_mul = 100
potential_mul = 10000
urban_year_region = (mask_province * year_mul + urban_arr).astype(int)
urban_potential_region = (mask_province * potential_mul + urban_potential).astype(int)


with ProgressBar():
    area_hist = da.bincount(urban_year_region.data.flatten(), weights=urban_area.data.flatten())
    area_potential = da.bincount(urban_potential_region.data.flatten(), weights=urban_area.data.flatten())
    area_hist, area_potential = dask.compute(area_hist, area_potential)



# Get the historical urban area
area_hist_df = pd.DataFrame(enumerate(area_hist), columns=['encode', 'Area_km2'])
area_hist_df['Province_code'] = area_hist_df['encode'] // year_mul
area_hist_df['year_code'] = area_hist_df['encode'] % year_mul

area_hist_df['Province'] = area_hist_df['Province_code'].map(dict(enumerate(UNIQUE_VALUES['Province'])))
area_hist_df['year'] = area_hist_df['year_code'].map({k+1:v for k, v in enumerate(UNIQUE_VALUES['Urban_map_year'][::-1])})

area_hist_df = area_hist_df.dropna()
area_hist_df['year'] = area_hist_df['year'].astype(int)

area_hist_df = area_hist_df.sort_values(['Province', 'year'])
area_hist_df['Area_cumsum_km2'] = area_hist_df.groupby('Province')['Area_km2'].cumsum()
area_hist_df = area_hist_df[['Province', 'year', 'Area_cumsum_km2']]


# Get the potential urban area
urban_area_potential = pd.DataFrame(enumerate(area_potential), columns=['encode', 'Area_km2'])
urban_area_potential['Province_code'] = urban_area_potential['encode'] // potential_mul
urban_area_potential['Potential'] = urban_area_potential['encode'] % potential_mul

urban_area_potential['Province'] = urban_area_potential['Province_code'].map(dict(enumerate(UNIQUE_VALUES['Province'])))
urban_area_potential = urban_area_potential.sort_values(['Province', 'Potential'], ascending=[True, False])

urban_area_potential = urban_area_potential.dropna().reset_index(drop=True)
urban_area_potential['Area_cumsum_km2'] = urban_area_potential.groupby('Province')['Area_km2'].cumsum()
urban_area_potential = urban_area_potential[['Province', 'Potential', 'Area_cumsum_km2']]



# Convert the list to a Dask array
area_hist_df.to_csv('data/results/step_8_area_hist_df.csv', index=False)
urban_area_potential.to_csv('data/results/step_8_urban_area_potential.csv', index=False)



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
    
    g.save('data/results/fig_step_8_1_urban_area_prediction_cumsum_km2.svg')
    
    g = (plotnine.ggplot(urban_area_potential.query('Potential > 9900')) +
        plotnine.aes(x='Potential', y='Area_cumsum_km2', color='Province') +
        plotnine.geom_line() +
        plotnine.theme_bw() +
        plotnine.theme(axis_text_x=plotnine.element_text(rotation=45, hjust=1)) +
        plotnine.scale_x_reverse()
        )
    
    g.save('data/results/fig_step_8_2_urban_area_prediction_under_different_transition_potential.svg')


