
import numpy as np
import pandas as pd
import rasterio

import rioxarray
import statsmodels.api as sm
import dask.array as da
import xarray

from helper_func.parameters import (BASE_YR, 
                                    TARGET_YR,
                                    PRED_STEP,
                                    DIM_ABBRIVATION, 
                                    UNIQUE_VALUES, 
                                    Monte_Carlo_num,
                                    Province_names_cn_en)


def compute_mean_std(paths: list[str]):
    """
    Compute the mean and standard deviation of multiple raster files.

    Parameters:
    paths (list[str]): A list of file paths to the raster files.

    Returns:
    tuple: A tuple containing the mean and standard deviation as xarray DataArrays.
    """

    tif_files = [rioxarray.open_rasterio(path).expand_dims({'model': [idx]}) 
                 for idx,path in enumerate(paths)]
    xr = xarray.combine_by_coords(tif_files)
    mean, std = xr.mean(dim='model'), xr.std(dim='model')
    return mean, std


# function to read yearbook csv and orginize data
def read_yearbook(path:str, record_name:str=None, city_cn_en:dict=Province_names_cn_en):

    # read and reshape data to long format
    df = pd.read_csv(path)
    df = df.set_index('地区')
    df = df.stack().reset_index()
    df.columns = ['Province','year','Value']
    df['year'] = df['year'].apply(lambda x: int(x[:4]))

    if record_name in {'Wheat', 'Rice', 'Maize'}:
        df['crop'] = record_name
    elif record_name in {'GDP', 'population'}:
        df['type'] = record_name

    # fitler df and replace CN to EN
    df = df[df['Province'].isin(city_cn_en.keys())]
    df = df.replace(city_cn_en)

    # remove 0s
    df = df[df['Value']!=0]
    return df.sort_values(['Province','year']).reset_index(drop=True)



def read_ssp(data_path:str='data/SSP_China_data'):
    SSP_GDP = pd.read_csv(f'{data_path}/SSP_GDP_ppp.csv')
    SSP_Pop = pd.read_csv(f'{data_path}/SSP_Population.csv')

    # get the shared cols
    same_cols = set(SSP_GDP.columns)&set(SSP_Pop.columns)
    same_cols = sorted(same_cols)
    same_cols.remove('2010.1')
    same_cols = ['Scenario'] + same_cols[:-1]

    SSP_GDP = SSP_GDP[same_cols]
    SSP_Pop = SSP_Pop[same_cols]

    # reshape df to long format
    SSP_GDP_long = SSP_GDP.set_index(['Scenario']).stack().reset_index()
    SSP_GDP_long.columns = ['Scenario','year','GDP']
    SSP_GDP_long['year'] = SSP_GDP_long['year'].astype('int16')

    SSP_Pop_long = SSP_Pop.set_index(['Scenario']).stack().reset_index()
    SSP_Pop_long.columns = ['Scenario','year','Population']
    SSP_Pop_long['year'] = SSP_Pop_long['year'].astype('int16')
    
    return SSP_GDP_long, SSP_Pop_long





# Function to convert a numpy array to a pandas dataframe
def ndarray_to_df(in_array:np.ndarray, in_dim:str, year_start:int=2020):
    
    # Check if "y" included in the in_dim
    if 'y' in in_dim:
        if year_start == 2010:
            DIM_ABBRIVATION['y'] = 'attainable_year'
        elif year_start == 2020:
            DIM_ABBRIVATION['y'] = 'simulation_year'
        
    in_names = [DIM_ABBRIVATION[i] for i in in_dim]
    
    out_df = pd.DataFrame(in_array.flatten(), 
                          index=pd.MultiIndex.from_product([UNIQUE_VALUES[i] for i in in_names])).reset_index()
    
    out_df.columns = in_names + ['Value']
    
    return out_df


def fit_linear_model(df):
    
    # Fit a linear model to the data
    X = df['year']
    y = df['Yield (tonnes)']
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    
    # Extrapolate the model to from <BASE_YR> to <TARGET_YR>
    pred_years = pd.DataFrame({'year': range(BASE_YR, TARGET_YR + 1, PRED_STEP)})
    pred_years = sm.add_constant(pred_years)
    
    extrapolate_df = model.get_prediction(pred_years)
    extrapolate_df = extrapolate_df.summary_frame(alpha=0.05)
    
    extrapolate_df['std'] = (extrapolate_df['obs_ci_upper'] - extrapolate_df['obs_ci_lower']) / (2 * 1.96)
    extrapolate_df['year'] = pred_years['year']
    extrapolate_df = extrapolate_df[['year','mean', 'std','obs_ci_upper','obs_ci_lower']]


    return extrapolate_df



def sample_from_mean_std(mean: np.ndarray, std: np.ndarray, resfactor: int = 8, size: int = Monte_Carlo_num):
    
    if mean.shape != std.shape:
        raise ValueError("The mean and std arrays must have the same shape")

    # Expand the mean and std arrays with an extra dimension for the samples
    mean = np.expand_dims(mean, 0).astype(np.float16)
    std = np.expand_dims(std, 0).astype(np.float16)

    # Get the chunk shape
    chunk_shape = (1,) + mean.shape[1:-2] + tuple(dim//resfactor for dim in mean.shape[-2:])
    sample_shape = (size,) + mean.shape[1:]

    # Create dask arrays
    mean_da = da.from_array(mean, chunks=chunk_shape)
    std_da = da.from_array(std, chunks=chunk_shape)

    # Generate samples
    sample_arr = da.random.normal(mean_da, std_da, sample_shape).astype(np.float16)

    # Rechunk sample_arr to have the same chunk shape as mean_da and std_da
    sample_arr = sample_arr.rechunk(chunk_shape)

    return sample_arr