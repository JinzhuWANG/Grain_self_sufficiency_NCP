
import numpy as np
import pandas as pd
import rasterio

from helper_func.parameters import DIM_ABBRIVATION, UNIQUE_VALUES, Province_names_cn_en


def compute_mean_std(paths:list[str]):
    """
    Compute the mean and std of the attainable yield for the given paths
    """
    # Read the tif files
    tif_files = [rasterio.open(path) for path in paths]
    # Compute the mean and std, replace the nodata value with 0
    concat_arr = np.stack([np.where(tif.read(1) == tif.nodata, 0, tif.read(1)) 
                           for tif in tif_files], axis=0)
    # Change negative values to 0
    concat_arr[concat_arr < 0] = 0
    
    # Compute the mean and std
    mean = np.mean(concat_arr, axis=0)
    std = np.std(concat_arr, axis=0)
    return mean, std


def extrapolate_array(idx:tuple, in_df:pd.DataFrame, group_vars:list[str]):
    # Sort the df by year 2025, 2055, 2085
    df = in_df.sort_values('year')
    
    # Linearly interpolated tif paths
    step_2025_2055 = (df.iloc[1]['mean'] - df.iloc[0]['mean'])/ (2055 - 2025)
    step_2055_2085 = (df.iloc[2]['mean'] - df.iloc[1]['mean'])/ (2085 - 2055)
    
    start_2010 = df.iloc[0]['mean'] - step_2025_2055 * (2025 - 2010)
    mid_2055 = df.iloc[1]['mean']
    end_2100 = df.iloc[1]['mean'] + step_2055_2085 * (2100 - 2055)
    
    mean_2010_2055 = np.linspace(start_2010, mid_2055, int((2055 - 2010)/5 + 1))
    mean_2055_2100 = np.linspace(mid_2055, end_2100, int((2100 - 2055)/5 + 1))
    mean_2010_2100 = np.concatenate([mean_2010_2055, mean_2055_2100[1:]])
    
    std_2010_2055 = np.stack([df.iloc[1]['std']] * len(mean_2010_2055), axis=0)
    std_2055_2100 = np.stack([df.iloc[2]['std']] * len(mean_2055_2100), axis=0)
    std_2010_2100 = np.concatenate([std_2010_2055, std_2055_2100[1:]])

    
    # Save the interpolated mean and std
    stats = {**dict(zip(group_vars, idx)),
                'year': np.arange(2010, 2101, 5),
                'mean': mean_2010_2100, 
                'std': std_2010_2100, 
    }
    
    return stats


# function to read yearbook csv and orginize data
def read_yearbook(path:str, crop:str, city_cn_en:dict=Province_names_cn_en):

    # read and reshape data to long format
    df = pd.read_csv(path)
    df = df.set_index('地区')
    df = df.stack().reset_index()
    df.columns = ['Province','year','Value']
    
    df['year'] = df['year'].apply(lambda x: int(x[:4]))
    df['crop'] = crop

    # fitler df and replace CN to EN
    df = df[df['Province'].isin(city_cn_en.keys())]
    df = df.replace(city_cn_en)

    # remove 0s
    df = df[df['Value']!=0]

    return df


# Function to convert a numpy array to a pandas dataframe
def ndarray_to_df(in_array:np.ndarray, in_dim:str):
    in_names = [DIM_ABBRIVATION[i] for i in in_dim]
    out_df = pd.DataFrame(in_array.flatten(), 
                          index=pd.MultiIndex.from_product([UNIQUE_VALUES[i] for i in in_names])).reset_index()
    out_df.columns = in_names + ['Value']
    return out_df



