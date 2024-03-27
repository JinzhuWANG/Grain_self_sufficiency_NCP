import numpy as np
import pandas as pd
import rasterio
from helper_func.parameters import Unique_values



# Define the columns for ordering the data
sort_volumns_GYGA = ['Province', 'crop','water_supply',]                    # p, c, s
sort_columns_GAEZ = ['rcp', 'crop', 'water_supply', 'c02_fertilization'	]   # r, c, s, o

# Get the number of the unique values for each columns 
unique_val_GAEZ = [len(Unique_values[i]) for i in sort_columns_GAEZ]
unique_val_GYGA = [len(Unique_values[i]) for i in sort_volumns_GYGA]



# Read the GYGA attainable yield data
GYGA_PY = pd.read_csv('data/GYGA/GYGA_attainable_filled.csv').replace('Rainfed', 'Dryland')
GYGA_PY_2010 = GYGA_PY.groupby(['Province','crop','water_supply']).mean(numeric_only=True).reset_index()
GYGA_PY_2010 = GYGA_PY_2010.sort_values(by=sort_volumns_GYGA).reset_index(drop=True)

GYGA_PY_2010 = GYGA_PY_2010['yield_potential'].values.reshape(*unique_val_GYGA)                            # (p, c, s) province, crop, water_supply


# Read the GAEZ attainable yield data
GAEZ_PY = pd.read_pickle('data/results/GAEZ_attainable.pkl')
GAEZ_PY = GAEZ_PY.sort_values(by=sort_columns_GAEZ).reset_index(drop=True)
# Only use array.shape for fast viewing the df
GAEZ_PY_repr = GAEZ_PY.map(lambda x: x.shape if isinstance(x, np.ndarray) and len(x.shape) > 1 else x)  


# Get the GYGA attainable yield data for 2010 
GAEZ_PY_2010 = GAEZ_PY['mean'].apply(lambda x: x[0]).tolist()                                               # r*(h, w)   
GAEZ_arr_shape = list(GAEZ_PY_2010[0].shape)                                                                # (h, w) 
GAEZ_PY_2010 = np.stack(GAEZ_PY_2010, axis=0).flatten().reshape(*(unique_val_GAEZ + GAEZ_arr_shape))        # (r, c, s, o, h, w) 


# Read the mean mask
with rasterio.open('data/GAEZ_v4/Province_mask_mean.tif') as src:
    mask_mean = src.read()                                          # (p, h, w) 


# Compute the mean GAEZ attainable yield for 2010
GAEZ_PY_2010_mean = np.einsum('rcsohw,phw->rcsop', GAEZ_PY_2010, mask_mean)       # (r, c, s, o, p) 


# Rearrange the dimensions of GAEZ_PY_2010_mean to match GYGA_PY_2010
GAEZ_PY_2010_mean = np.transpose(GAEZ_PY_2010_mean, (4, 1, 2, 0, 3))            # (p, c, s, r, o)

# Now you can subtract GAEZ_PY_2010_mean from GYGA_PY_2010 for each r and o
diff_PY_2010 = GYGA_PY_2010[:, :, :, None, None] - GAEZ_PY_2010_mean            # (p, c, s, r, o)


# Spred the diff_PY_2010 to the mask_mean
diff_PY_2010 = np.einsum('pcsro,phw->pcsrohw', diff_PY_2010, mask_mean)         # (p, c, s, r, o, h, w)

