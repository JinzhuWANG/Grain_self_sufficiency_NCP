
from asyncio import tasks
import numpy as np
import pandas as pd

import rioxarray
import statsmodels.api as sm
import dask.array as da
import xarray

from joblib import Parallel, delayed
from tqdm.auto import tqdm
from scipy.stats import norm, truncnorm

from helper_func.parameters import (BASE_YR, 
                                    TARGET_YR,
                                    PRED_STEP,
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

    if record_name in {'Wheat', 'Wetland rice', 'Maize'}:
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


def sample_ppf(mean, std, n_samples=100, seed=0):
    np.random.seed(seed)
    
    # Get a uniform distribution between 0.001 and 0.999
    samples = np.random.uniform(0.001, 0.999, n_samples)
    
    # Use a consistent random seed within the parallel execution
    def calculate_ppf(x):
        np.random.seed(seed)  # Ensure the seed is set for each worker
        return norm.ppf(x, loc=mean, scale=std)
    
    tasks = [delayed(calculate_ppf)(n) for n in samples]
    para_obj = Parallel(n_jobs=-1, prefer='threads', return_as='generator')
    
    ppf_xr = []
    for res in tqdm(para_obj(tasks), total=n_samples):
        ppf_xr.append(res.astype('float32'))
    
    return da.from_array(ppf_xr)