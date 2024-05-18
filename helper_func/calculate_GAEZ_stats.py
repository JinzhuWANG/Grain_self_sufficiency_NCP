import itertools
import numpy as np
import pandas as pd
import rioxarray
import xarray as xr

from helper_func.parameters import UNIQUE_VALUES, GAEZ_variables, GAEZ_water_supply


def get_GAEZ_df(in_path:str = 'data/GAEZ_v4/GAEZ_df.csv', var_type:str = 'Harvested area'):
    # Read the GAEZ_df which records the metadata of the GAEZ data path
    GAEZ_df = pd.read_csv(in_path)
    GAEZ_df = GAEZ_df.query('GAEZ == "GAEZ_5" and variable == @var_type and water_supply != "Total"')
    GAEZ_df = GAEZ_df.replace(GAEZ_water_supply['GAEZ_5']).infer_objects(copy=False)
    GAEZ_df = GAEZ_df[GAEZ_variables['GAEZ_5'] + ['fpath']]
    GAEZ_df = GAEZ_df.sort_values(by=['crop', 'water_supply']).reset_index(drop=True)                     
    return GAEZ_df
        


def get_GEAZ_layers(GAEZ_df:pd.DataFrame):
    
    GAEZ_xr = [rioxarray.open_rasterio(fpath) for fpath in GAEZ_df['fpath']]
    
    GAEZ_meta = GAEZ_xr[0]
    GAEZ_arr = np.array([i.values for i in GAEZ_xr])
    GAEZ_arr = GAEZ_arr.reshape(
        GAEZ_df['crop'].nunique(),
        GAEZ_df['water_supply'].nunique(), 
        GAEZ_meta.shape[-2],
        GAEZ_meta.shape[-1]) 

    return xr.DataArray(
        GAEZ_arr, 
        coords={
            'crop': GAEZ_df['crop'].unique(),
            'water_supply': GAEZ_df['water_supply'].unique(),
            'y': GAEZ_meta.y.values,
            'x': GAEZ_meta.x.values
        })



def get_GAEZ_stats(var_type:str):
    """
    Calculate statistics for a given variable type (either 'Harvested area' or 'Yield') based on GAEZ data.

    Parameters:
    var_type (str): The type of variable to calculate statistics for. Must be one of ['Harvested area', 'Yield'].

    Returns:
    pandas.DataFrame: A DataFrame containing the calculated statistics for each province.

    Raises:
    ValueError: If var_type is not one of ['Harvested area', 'Yield'].
    """

    if var_type not in ['Harvested area', 'Yield']:
        raise ValueError('variable must be either "Harvested area" or "Yield"')

    GAEZ_df = get_GAEZ_df(var_type = var_type)
    GAEZ_xr = get_GEAZ_layers(GAEZ_df)

    mask_mean = rioxarray.open_rasterio('data/GAEZ_v4/Province_mask_mean.tif')
    mask_sum = rioxarray.open_rasterio('data/GAEZ_v4/Province_mask.tif')
    mask_sum = mask_sum.where(mask_sum>0, 0)  # Negative values are not allowed for bincount

    # Yield needs to be weighted by 1/total_pixels, so to get the mean value
    GAEZ_xr = GAEZ_xr * mask_mean if var_type == 'Yield' else GAEZ_xr

    # Loop through each dimension besides x and y
    dims = [i for i in GAEZ_xr.dims if i not in ['x', 'y']]
    dim_val = list(itertools.product(*[GAEZ_xr[i].values for i in dims]))

    stats_dfs = []
    for coord in dim_val:
        sel_dict = dict(zip(dims, coord))
        stats = np.bincount(mask_sum.values.flatten(), weights=GAEZ_xr.sel(**sel_dict).values.flatten())
        stats = dict(zip(UNIQUE_VALUES['Province'], stats))
        sel_dict.update(**stats)
        stats_dfs.append(sel_dict)
        
    # Convert to DataFrame
    stats_df = pd.DataFrame(stats_dfs)
    stats_df = stats_df.set_index(dims).stack().reset_index()
    stats_df.columns = dims + ['Province', var_type]

    return stats_df