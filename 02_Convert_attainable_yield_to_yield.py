import numpy as np
import pandas as pd
import rasterio
import plotnine

from helper_func import read_yearbook
from helper_func.parameters import (Attainable_conversion, 
                                    Projection_years, Province_names_cn_en, 
                                    Province_names_en,
                                    )

# Read the GAEZ_extrapolated_df which records the extrapolated attainable yield between 2020-2100
GAEZ_attainable = pd.read_pickle('data/results/GAEZ_4_extrapolated.pkl')

# Multiply the attainable yield by the conversion factor
GAEZ_attainable['mean'] = GAEZ_attainable.apply(lambda x: x['mean'] * Attainable_conversion[x['crop']], axis=1)
GAEZ_attainable['std'] = GAEZ_attainable.apply(lambda x: x['std'] * Attainable_conversion[x['crop']], axis=1)
GAEZ_attainable.to_pickle('data/results/GAEZ_attainable.pkl')




# Read the Province_mask_mean for computing the mean statistics for each province
with rasterio.open('data/GAEZ_v4/Province_mask_mean.tif') as src:
    Province_mask_mean_phw = src.read()              # (province, height, width)

# Compute the mean and std of attainable yield for each province
mean_ryhw = np.stack(GAEZ_attainable['mean']) # (row, year, height, width)
std_ryhw = np.stack(GAEZ_attainable['std'])   # (row, year, height, width)

# Compute the yield for each province
yield_rpy = np.einsum('phw,ryhw->rpy', Province_mask_mean_phw, mean_ryhw) # (row, province, year)



# Create a DataFrame for the production
yield_df = GAEZ_attainable[['rcp', 'crop', 'water_supply', 'c02_fertilization']].copy()
yield_df['Province'] = [Province_names_en] * len(yield_df)
yield_df['Year'] = [Projection_years] * len(yield_df)
yield_df = yield_df.explode('Province').explode('Year').reset_index(drop=True)
yield_df['Year'] = yield_df['Year'].astype(int)
yield_df['Production (kg)'] = yield_rpy.flatten()
yield_df['Production (t)'] = yield_df['Production (kg)'] / 1000


# Read the yearbook data
wheat_yield_history = read_yearbook('data/Yearbook/Provincial_wheat_yield.csv','Wheat')
rice_yield_history = read_yearbook('data/Yearbook/Provincial_rice_yield.csv','Wetland rice')
maize_yield_history = read_yearbook('data/Yearbook/Provincial_maize_yield.csv','Maize')

yield_yearbook = pd.concat([wheat_yield_history, rice_yield_history, maize_yield_history], axis=0)



# Plot the production for each province of both yearbook and GAEZ
plotnine.options.figure_size = (16, 6)
plotnine.options.dpi = 100
g = (
    plotnine.ggplot() +
    plotnine.geom_line(yield_df.query('rcp == "RCP2.6" and c02_fertilization == "With CO2 Fertilization"'), 
                       plotnine.aes(x='Year', y='Production (kg)', color='water_supply')) +
    plotnine.geom_point(yield_yearbook, plotnine.aes(x='year', y='Value')) +
    plotnine.facet_grid('crop~Province', scales='free_y') +
    plotnine.theme_minimal() +
    plotnine.theme(axis_text_x=plotnine.element_text(rotation=45, hjust=1))
)


