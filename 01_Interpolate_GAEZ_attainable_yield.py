import pandas as pd
import rasterio
import geopandas as gpd
import matplotlib.pyplot as plt

from helper_func.parameters import GAEZ_variables



# Read the GAEZ_df which records the metadata of the GAEZ data
GAEZ_df = pd.read_csv('data/GAEZ_v4/GAEZ_df.csv')
GAEZ_4_df = GAEZ_df.query('GAEZ == "GAEZ_4"')

# Compute the mean and std of attainable yield according to different climate models
group_vars = GAEZ_variables['GAEZ_4'].copy()
group_vars.remove('model')

# Read the shp of the research region
research_region_shp = gpd.read_file('data/Vector_boundary/North_china_Plain_Province.shp')



src = rasterio.open(GAEZ_4_df.iloc[0]['fpath'])
src_arr = src.read(1)