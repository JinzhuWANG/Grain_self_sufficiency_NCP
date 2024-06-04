import numpy as np
import pandas as pd
import rioxarray
import xarray as xr

from helper_func.parameters import UNIQUE_VALUES, GAEZ_variables, GAEZ_water_supply

# Get the GAEZ masks
mask_mean = rioxarray.open_rasterio('data/GAEZ_v4/Province_mask_mean.tif')
mask_sum = rioxarray.open_rasterio('data/GAEZ_v4/Province_mask.tif')
mask_sum = mask_sum.where(mask_sum>0, 0)  # Negative values are not allowed for bincount

def weighted_bincount(mask, weights, minlength=None):
    return np.bincount(mask.ravel(), weights=weights.ravel(), minlength=minlength)


def get_GAEZ_df(in_path:str = 'data/GAEZ_v4/GAEZ_df.csv', var_type:str = 'Harvested area'):
    # Read the GAEZ_df which records the metadata of the GAEZ data path
    GAEZ_df = pd.read_csv(in_path)
    GAEZ_df = GAEZ_df.query('GAEZ == "GAEZ_5" and variable == @var_type and water_supply != "Total"')
    GAEZ_df = GAEZ_df.replace(GAEZ_water_supply['GAEZ_5']).infer_objects(copy=False)
    GAEZ_df = GAEZ_df[GAEZ_variables['GAEZ_5'] + ['fpath']]
    GAEZ_df = GAEZ_df.sort_values(by=['crop', 'water_supply']).reset_index(drop=True)                     
    return GAEZ_df
        


def get_GEAZ_layers(GAEZ_df: pd.DataFrame):
    """
    Combines multiple GeoTIFF layers into a single xarray dataset.

    Args:
        GAEZ_df (pd.DataFrame): A DataFrame containing information about the GeoTIFF layers.
            It should have the following columns: 'fpath', 'crop', 'water_supply'.

    Returns:
        xr.Dataset: A combined xarray dataset containing all the GeoTIFF layers.

    """
    xr_arrs = []
    for _, row in GAEZ_df.iterrows():
        path = row['fpath']
        crop = row['crop']
        water_supply = row['water_supply']

        xr_arr = rioxarray.open_rasterio(path).squeeze()
        xr_arr = xr_arr.expand_dims({'crop': [crop], 'water_supply': [water_supply]})
        xr_arrs.append(xr_arr)
        
    return xr.combine_by_coords(xr_arrs)



def bincount_with_mask(mask, xr_arr):
    """
    Calculate statistics using weighted bincount with a mask.

    Parameters:
    - mask (xarray.DataArray): The mask array.
    - xr_arr (xarray.DataArray): The input array.

    Returns:
    - stats (pandas.DataFrame): The calculated statistics.

    """
    stats = xr.apply_ufunc(
        weighted_bincount,
        mask,
        xr_arr,
        input_core_dims=[['y', 'x'], ['y', 'x']],
        output_core_dims=[['bin']],
        vectorize=True,
        dask='parallelized',
        kwargs={'minlength': int(mask_sum.max().values) + 1},
        output_dtypes=[float],
        dask_gufunc_kwargs={'output_sizes': {'bin': int(mask.max().values) + 1}}
    )
    
    stats.name = 'Value'
    stats = stats.to_dataframe().reset_index()
    return stats



def get_GAEZ_stats(var_type:str):
    """
    Calculate statistics for a given variable type (either "Harvested area" or "Yield") using GAEZ data.

    Parameters:
    var_type (str): The type of variable to calculate statistics for. Must be either "Harvested area" or "Yield".

    Returns:
    pandas.DataFrame: A DataFrame containing the calculated statistics for each province.

    Raises:
    ValueError: If the var_type is not "Harvested area" or "Yield".
    """

    if var_type not in ['Harvested area', 'Yield']:
        raise ValueError('variable must be either "Harvested area" or "Yield"')

    GAEZ_df = get_GAEZ_df(var_type = var_type)
    GAEZ_xr = get_GEAZ_layers(GAEZ_df)
    GAEZ_xr = GAEZ_xr * mask_mean if var_type == 'Yield' else GAEZ_xr

    GAEZ_stats = bincount_with_mask(mask_sum, GAEZ_xr)
    GAEZ_stats.rename(columns={'Value': var_type, 'bin':'Province'}, inplace=True)
    GAEZ_stats['Province'] = GAEZ_stats['Province'].map(dict(enumerate(UNIQUE_VALUES['Province'])))
    
        
    return GAEZ_stats[['Province', 'crop', 'water_supply', var_type]]