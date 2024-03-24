import numpy as np
import pandas as pd
import concurrent
import concurrent.futures
import geopandas as gpd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from helper_func import compute_mean_std, extrapolate_array
from helper_func.parameters import GAEZ_variables, GAEZ_year_mid

# Compute the mean and std of attainable yield according to different climate models
group_vars = GAEZ_variables['GAEZ_4'].copy()
group_vars.remove('model')


# Read the GAEZ_df which records the metadata of the GAEZ data
GAEZ_df = pd.read_csv('data/GAEZ_v4/GAEZ_df.csv').replace(GAEZ_year_mid)    # Replace the year with the mid year
GAEZ_4_df = GAEZ_df.query('GAEZ == "GAEZ_4" and year != "1981-2010"')       # Remove historical data




# Group by the group_vars and compute the mean and std of attainable yield
dfs = []
for idx, df in list(GAEZ_4_df.groupby(group_vars)):
    # Get tif paths
    tif_paths = df['fpath'].tolist()
    # Compute the mean and std for all tifs
    mean, std = compute_mean_std(tif_paths)
    
    dfs.append({**dict(zip(group_vars, idx)),
                'mean': mean, 
                'std': std, 
                })
    
# Concatenate the results
GAEZ_4_df = pd.DataFrame(dfs)
    



# Remove year from the group_vars
group_vars.remove('year')

dfs = []
with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = []
    for idx, df in list(GAEZ_4_df.groupby(group_vars)):
        futures.append(executor.submit(extrapolate_array, idx, df, group_vars))

    for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
        try:
            dfs.append(future.result())
        except Exception as e:
            print(e)

    
# Concatenate the results
GAEZ_4_df = pd.DataFrame(dfs)

# Save the results
GAEZ_4_df.to_pickle('data/results/GAEZ_4_extrapolated.pkl')