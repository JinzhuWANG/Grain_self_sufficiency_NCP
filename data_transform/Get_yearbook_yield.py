import pandas as pd
from helper_func import read_yearbook



# Read the yearbook data
wheat_yield_history = read_yearbook('data/Yearbook/Provincial_wheat_yield.csv','Wheat')
rice_yield_history = read_yearbook('data/Yearbook/Provincial_rice_yield.csv','Wetland rice')
maize_yield_history = read_yearbook('data/Yearbook/Provincial_maize_yield.csv','Maize')

# Concatenate the data, and convert kg to tonnes
yield_yearbook = pd.concat([wheat_yield_history, rice_yield_history, maize_yield_history], axis=0)
yield_yearbook['Yield (tonnes)'] = yield_yearbook['Value'] / 1000

# Save the yield_yearbook
yield_yearbook.to_csv('data/results/yield_yearbook.csv', index=False)

