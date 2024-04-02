import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from helper_func import compute_mean_std, extrapolate_array
from helper_func.parameters import GAEZ_variables, GAEZ_year_mid, UNIQUE_VALUES

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
for idx, df in tqdm(list(GAEZ_4_df.groupby(group_vars))):
    result = extrapolate_array(idx, df, group_vars)
    dfs.append(result)




# Concatenate the results
GAEZ_4_df = pd.DataFrame(dfs)
# Sort the df by the group_vars
GAEZ_4_df = GAEZ_4_df.sort_values(by=group_vars).reset_index(drop=True)
# Only use array.shape for fast viewing the df
GAEZ_4_df_repr = GAEZ_4_df.map(lambda x: x.shape if isinstance(x, np.ndarray) and len(x.shape) > 1 else x)

# Get the height and width of the array
array_shape = list(GAEZ_4_df['mean'][0].shape[1:])
# Get the length of the group_vars
len_group_vars = [len(UNIQUE_VALUES[i]) for i in group_vars]  # (r, c, s, o)
# Get the complete shape of the array
complete_shape = len_group_vars + [len(UNIQUE_VALUES['attainable_year'])] + array_shape                 # (r, c, s, o, y, h, w)

# Get the numpy array of the mean and std
GAEZ_4_array_mean = np.stack(GAEZ_4_df['mean']).flatten().reshape(*complete_shape).astype(np.float16)  # (r, c, s, o, y, h, w)
GAEZ_4_array_std = np.stack(GAEZ_4_df['std']).flatten().reshape(*complete_shape).astype(np.float16)    # (r, c, s, o, y, h, w)  

# The std of Wetland rice in Dryland should be 0
GAEZ_4_array_std[:,
                 UNIQUE_VALUES['crop'].index('Wetland rice'),
                 UNIQUE_VALUES['water_supply'].index('Dryland'),
                 :,
                 :,
                 :,
                 :,] = 0

# Save the results
np.save('data/results/GAEZ_4_attain_extrapolated_mean_rcsoyhw.npy', GAEZ_4_array_mean)
np.save('data/results/GAEZ_4_attain_extrapolated_std_rcsoyhw.npy', GAEZ_4_array_std)

