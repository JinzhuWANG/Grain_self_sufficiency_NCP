import numpy as np
import pandas as pd
import dask
import dask.array as da
import h5py
import plotnine

from helper_func.parameters import HDF_BLOCK_SIZE, UNIQUE_VALUES, Province_names_cn_en
from dask.diagnostics import ProgressBar


# Read the lucc area
lucc_area_ds = h5py.File('data/LUCC/LUCC_area_km2.hdf5', 'r')
lucc_area_area = da.from_array(lucc_area_ds['area'], chunks=lucc_area_ds['area'].chunks) 


# Read the urban data
urban_ds = h5py.File('data/LUCC/Urban_1990_2019.hdf5', 'r')
urban_arr = da.from_array(urban_ds['Array'], chunks=urban_ds['Array'].chunks)[0]


with dask.config.set(**{'array.slicing.split_large_chunks': True}), ProgressBar():
    # Compute the bincount for the province and append it to the list
    area = da.bincount(urban_arr.ravel(), weights=lucc_area_area.ravel()).compute()









# Convert the list to a Dask array
counts_per_province = pd.DataFrame(counts_per_province[:, 1:], 
                                   index=Province_names_cn_en.values(), 
                                   columns=UNIQUE_VALUES['Urban_map_year'])

# Transpose the dataframe, so that the years are the columns
counts_per_province = counts_per_province.T.reset_index().rename(columns={'index':'Year'})
counts_per_province = counts_per_province.melt(id_vars='Year', var_name='Province', value_name='Urban_Area')
counts_per_province['Urban_count'] = counts_per_province.groupby('Province')['Urban_Area'].transform('cumsum')

counts_per_province.to_csv('data/results/Urban_1990_2019_count.csv', index=False)



# Sanity check
if __name__ == '__main__':
    plotnine.options.figure_size = (12, 8)
    plotnine.options.dpi = 100
    
    g = (plotnine.ggplot(counts_per_province) +
        plotnine.aes(x='Year', y='Urban_count', color='Province') +
        plotnine.geom_line() +
        plotnine.theme_bw() +
        plotnine.theme(axis_text_x=plotnine.element_text(rotation=45, hjust=1))
        )


