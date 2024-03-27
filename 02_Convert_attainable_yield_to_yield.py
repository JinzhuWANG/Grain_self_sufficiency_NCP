import numpy as np
import pandas as pd
import rasterio
import plotnine

from helper_func import read_yearbook
from helper_func.parameters import (Attainable_conversion, 
                                    Projection_years, 
                                    Unique_values,
                                    )

# Get the convertion factor for each crop <dry weight -> kg harvested>
convesion_factor = np.array(list(Attainable_conversion.values()))                   # (c)


# Read the GAEZ_extrapolated_df which records the extrapolated attainable yield 
GAEZ_4_attain_extrapolated_mean_rcsoyhw = np.load('data/results/GAEZ_4_attain_extrapolated_mean_rcsoyhw.npy')
GAEZ_4_attain_extrapolated_std_rcsoyhw  = np.load('data/results/GAEZ_4_attain_extrapolated_std_rcsoyhw.npy')

# Multiply the attainable yield by the conversion factor
GAEZ_4_attain_extrapolated_mean_kg_rcsoyhw =  np.einsum('c,rcsoyhw->rcsoyhw', convesion_factor, GAEZ_4_attain_extrapolated_mean_rcsoyhw).astype(np.float16) 
GAEZ_4_attain_extrapolated_std_kg_rcsoyhw =  np.einsum('c,rcsoyhw->rcsoyhw', convesion_factor, GAEZ_4_attain_extrapolated_std_rcsoyhw).astype(np.float16)

# Save the results
np.save('data/results/GAEZ_4_attain_extrapolated_mean_kg_rcsoyhw.npy', GAEZ_4_attain_extrapolated_mean_kg_rcsoyhw)
np.save('data/results/GAEZ_4_attain_extrapolated_std_kg_rcsoyhw.npy', GAEZ_4_attain_extrapolated_std_kg_rcsoyhw)




# Read the Province_mask_mean for computing the mean statistics for each province
with rasterio.open('data/GAEZ_v4/Province_mask_mean.tif') as src:
    Province_mask_mean_phw = src.read()                                             # (province, h, w)



# Compute the mean and std of attainable yield for each province
GAEZ_4_attain_extrapolated_mean_kg_rcsopy = np.einsum('rcsoyhw,phw->rcsopy', 
                                                        GAEZ_4_attain_extrapolated_mean_kg_rcsoyhw, 
                                                        Province_mask_mean_phw)                         # (r, c, s, o, p, y)




# Read the yearbook data
wheat_yield_history = read_yearbook('data/Yearbook/Provincial_wheat_yield.csv','Wheat')
rice_yield_history = read_yearbook('data/Yearbook/Provincial_rice_yield.csv','Wetland rice')
maize_yield_history = read_yearbook('data/Yearbook/Provincial_maize_yield.csv','Maize')

yield_yearbook = pd.concat([wheat_yield_history, rice_yield_history, maize_yield_history], axis=0)


# Filter the yield_array with specific rcp, c02_fertilization, and water_supply
rcp = "RCP2.6" 
c02_fertilization = "With CO2 Fertilization"

GAEZ_4_attain_extrapolated_mean_kg_cspy = GAEZ_4_attain_extrapolated_mean_kg_rcsopy[Unique_values['rcp'].index(rcp), 
                                                                                            :, 
                                                                                            :, 
                                                                                            Unique_values['c02_fertilization'].index(c02_fertilization), 
                                                                                            :, 
                                                                                            :]


GAEZ_4_attain_extrapolated_mean_kg_cspy_df = pd.DataFrame(GAEZ_4_attain_extrapolated_mean_kg_cspy.flatten(),
                                                                index=pd.MultiIndex.from_product([Unique_values['crop'], 
                                                                                                    Unique_values['water_supply'],
                                                                                                    Unique_values['Province'],
                                                                                                    Projection_years,
                                                                                                    ]),
).reset_index().rename(columns={0:'Production (kg)', 'level_0':'crop', 'level_1':'water_supply', 'level_2':'Province', 'level_3':'Year'})



# Plot the production for each province of both yearbook and GAEZ
plotnine.options.figure_size = (16, 6)
plotnine.options.dpi = 100
g = (
    plotnine.ggplot() +
    plotnine.geom_line(GAEZ_4_attain_extrapolated_mean_kg_cspy_df,
                       plotnine.aes(x='Year', y='Production (kg)', color='water_supply')) +
    plotnine.geom_point(yield_yearbook, plotnine.aes(x='year', y='Value')) +
    plotnine.facet_grid('crop~Province', scales='free_y') +
    plotnine.theme_minimal() +
    plotnine.theme(axis_text_x=plotnine.element_text(rotation=45, hjust=1))
)


