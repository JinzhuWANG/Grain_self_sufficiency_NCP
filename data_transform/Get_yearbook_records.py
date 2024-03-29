import numpy as np
import pandas as pd
from helper_func import read_yearbook
from helper_func.parameters import BASE_YR, UNIQUE_VALUES



# Read the yearbook data
wheat_yield_history = read_yearbook('data/Yearbook/Provincial_wheat_yield.csv','Wheat')
rice_yield_history = read_yearbook('data/Yearbook/Provincial_rice_yield.csv','Wetland rice')
maize_yield_history = read_yearbook('data/Yearbook/Provincial_maize_yield.csv','Maize')

# Concatenate the data, and convert kg to tonnes
yearbook_yield = pd.concat([wheat_yield_history, rice_yield_history, maize_yield_history], axis=0)
yearbook_yield['Yield (tonnes)'] = yearbook_yield['Value'] / 1000

# Get the array format of the yearbook data
yearbook_yield_targ = yearbook_yield.query(f'year == {BASE_YR}')
yearbook_yield_targ_long = yearbook_yield_targ['Yield (tonnes)'].values
yearbook_yield_targ_pc = yearbook_yield_targ_long.reshape(len(UNIQUE_VALUES['Province']),len(UNIQUE_VALUES['crop']))        # (p, c)

# Save the yearbook data to disk
np.save('data/results/yearbook_yield_targ_pc.npy', yearbook_yield_targ_pc)
yearbook_yield.to_csv('data/results/yearbook_yield.csv', index=False)








# Read the yearbook data
wheat_area_history = read_yearbook('data/Yearbook/Area_wheat.csv','Wheat')
rice_area_history = read_yearbook('data/Yearbook/Area_rice.csv','Wetland rice')
maize_area_history = read_yearbook('data/Yearbook/Area_maize.csv','Maize')

# Concatenate the data, and convert ha to kha
yearbook_area = pd.concat([wheat_area_history, rice_area_history, maize_area_history], axis=0)
yearbook_area = yearbook_area.rename(columns={'Value':'area_yearbook_kha'})


yearbook_area_targ = yearbook_area.query(f'year == {BASE_YR}')
yearbook_area_targ_long = yearbook_area_targ['area_yearbook_kha'].values
yearbook_area_targ_pc = yearbook_area_targ_long.reshape(len(UNIQUE_VALUES['Province']),len(UNIQUE_VALUES['crop']))          # (p, c)

# save yearbook_cropland_area to disk
np.save('data/results/yearbook_area_targ_pc.npy', yearbook_area_targ_pc)
yearbook_area.to_csv('data/results/yearbook_area.csv',index=False)
