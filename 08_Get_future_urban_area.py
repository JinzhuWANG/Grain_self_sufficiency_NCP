import numpy as np
import pandas as pd
import dask
import dask.array as da
import h5py
import plotnine

from helper_func.parameters import UNIQUE_VALUES
from dask.diagnostics import ProgressBar


# Read the lucc mask
lucc_ds = h5py.File('data/LUCC/LUCC_Province_mask.hdf5', 'r')
lucc_mask = da.from_array(lucc_ds['Array'], chunks=lucc_ds['Array'].chunks)


# Read the lucc area
lucc_area_ds = h5py.File('data/LUCC/LUCC_area_km2.hdf5', 'r')
lucc_area_area = da.from_array(lucc_area_ds['area'], chunks=lucc_area_ds['area'].chunks) 
lucc_area_area = lucc_area_area * lucc_mask


# Read the urban data
urban_ds = h5py.File('data/LUCC/Urban_1990_2019.hdf5', 'r')
urban_arr = da.from_array(urban_ds['Array'], chunks=urban_ds['Array'].chunks)[0]
urban_arr = urban_arr * lucc_mask


# Read the urban potential data
urban_potential_ds = h5py.File('data/LUCC/Transition_potential.hdf5', 'r')
urban_potential_arr = da.from_array(urban_potential_ds['Array'], chunks=urban_potential_ds['Array'].chunks)
urban_potential_arr = urban_potential_arr * lucc_mask


area_each_province = []
for idx, Province in enumerate(UNIQUE_VALUES['Province']):
    with dask.config.set(**{'array.slicing.split_large_chunks': True}), ProgressBar():
        print(f'Processing {Province}')
        # Compute the bincount for the province and append it to the list
        area_hist = da.bincount(urban_arr[idx].ravel(), weights=lucc_area_area[idx].ravel())
        area_potential = da.bincount(urban_potential_arr[idx].ravel(), weights=lucc_area_area[idx].ravel())
        area_hist, area_potential = dask.compute(area_hist, area_potential)
        area_each_province.append({'Province': Province, 'Area_hist': area_hist[1:], 'Area_potential': area_potential[1:]})


df_hists, df_potentials = [], []
for d in area_each_province:
    # Get the keys you want to retrieve
    key_hist = ['Province', 'Area_hist']
    key_potential = ['Province', 'Area_potential']
    # Crete the df
    df_hist = pd.DataFrame({key: d[key] for key in key_hist})
    df_potential = pd.DataFrame({key: d[key] for key in key_potential})
    # Add year to hist and pixel-val to potential
    df_hist['year'] = UNIQUE_VALUES['Urban_map_year']
    df_potential['pixel_val'] = np.arange(1, len(df_potential) + 1)
    # Cumsum the value
    df_hist['Area_cumsum_km2'] = df_hist['Area_hist'].cumsum()
    df_potential['Area_cumsum_km2'] = df_potential['Area_potential'].cumsum()
    # Append the df to the list
    df_hists.append(df_hist)
    df_potentials.append(df_potential)

# Concatenate the list to a single df
urban_area_hist = pd.concat(df_hists)
urban_area_potential = pd.concat(df_potentials)


# Convert the list to a Dask array
urban_area_hist.to_csv('data/results/urban_area_hist.csv', index=False)
urban_area_potential.to_csv('data/results/urban_area_potential.csv', index=False)

# Read the data
urban_area_hist = pd.read_csv('data/results/urban_area_hist.csv')
urban_area_potential = pd.read_csv('data/results/urban_area_potential.csv')

# Sanity check
if __name__ == '__main__':
    plotnine.options.figure_size = (12, 8)
    plotnine.options.dpi = 100
    
    g = (plotnine.ggplot(urban_area_hist) +
        plotnine.aes(x='year', y='Area_cumsum_km2', color='Province') +
        plotnine.geom_line() +
        plotnine.theme_bw() +
        plotnine.theme(axis_text_x=plotnine.element_text(rotation=45, hjust=1))
        )


