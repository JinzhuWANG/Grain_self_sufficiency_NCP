import numpy as np
import pandas as pd
import rasterio

from helper_func.parameters import GAEZ_variables, GAEZ_water_supply


# Read the yearbook yield data
yearbook_yield = pd.read_csv('data/results/yield_yearbook.csv')

# Read the GAEZ_df which records the metadata of the GAEZ data path
GAEZ_df = pd.read_csv('data/GAEZ_v4/GAEZ_df.csv')
GAEZ_5_2010 = GAEZ_df.query('GAEZ == "GAEZ_5" and year == "2010"')    
GAEZ_5_2010 = GAEZ_5_2010.replace(GAEZ_water_supply['GAEZ_5']).infer_objects(copy=False)[GAEZ_variables['GAEZ_5'] + ['fpath']]
GAEZ_5_2010 = GAEZ_5_2010.sort_values(by=['crop', 'water_supply']).reset_index(drop=True)

# Read the mask data
with rasterio.open('data/GAEZ_v4/Province_mask.tif') as src:
    mask = src.read(1)

# Read the yield data of <water_supply == Total>
yield_total = GAEZ_5_2010.query('water_supply == "Total"')['fpath'].tolist()
yield_total = np.stack([rasterio.open(fpath).read(1) for fpath in yield_total], 0) # (c, h, w)















