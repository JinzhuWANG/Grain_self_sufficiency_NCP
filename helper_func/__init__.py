
import numpy as np
import pandas as pd
import rasterio


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
    # Sort the df by year
    df = in_df.sort_values('year')
    
    # Linearly interpolated tif paths
    step_2020_2055 = (df.iloc[1]['mean'] - df.iloc[0]['mean'])/ (2055 - 2025)
    step_2055_2100 = (df.iloc[2]['mean'] - df.iloc[1]['mean'])/ (2085 - 2055)
    
    start_2020 = df.iloc[0]['mean'] - step_2020_2055 * (2025 - 2020)
    mid_2055 = df.iloc[1]['mean']
    end_2100 = df.iloc[1]['mean'] + step_2055_2100 * (2100 - 2055)
    
    mean_2020_2055 = np.linspace(start_2020, mid_2055, int((2055 - 2020)/5 + 1))
    mean_2055_2100 = np.linspace(mid_2055, end_2100, int((2100 - 2055)/5 + 1))
    mean_2020_2100 = np.concatenate([mean_2020_2055, mean_2055_2100[1:]])
    
    std_2020_2055 = np.stack([df.iloc[1]['std']] * len(mean_2020_2055), axis=0)
    std_2055_2100 = np.stack([df.iloc[2]['std']] * len(mean_2055_2100), axis=0)
    std_2020_2100 = np.concatenate([std_2020_2055, std_2055_2100[1:]])

    
    # Save the interpolated mean and std
    stats = {**dict(zip(group_vars, idx)),
                'year': np.arange(2020, 2101, 5),
                'mean': mean_2020_2100, 
                'std': std_2020_2100, 
    }
    
    return stats