import numpy as np
import pandas as pd
import rasterio
import plotnine

from helper_func import ndarray_to_df, read_yearbook
from helper_func.parameters import UNIQUE_VALUES, Attainable_conversion
                                    

# Get the convertion factor for each crop <dry weight -> kg harvested>
convesion_factor = np.array(list(Attainable_conversion.values()))                   # (c)


# Read the GAEZ_extrapolated_df which records the extrapolated attainable yield 
GAEZ_4_attain_extrapolated_mean_rcsoyhw = np.load('data/results/GAEZ_4_attain_extrapolated_mean_rcsoyhw.npy')
GAEZ_4_attain_extrapolated_std_rcsoyhw  = np.load('data/results/GAEZ_4_attain_extrapolated_std_rcsoyhw.npy')

# Multiply the attainable yield by the conversion factor
GAEZ_4_attain_extrapolated_mean_kg_rcsoyhw =  np.einsum('c,rcsoyhw->rcsoyhw', convesion_factor, GAEZ_4_attain_extrapolated_mean_rcsoyhw).astype(np.float16) 
GAEZ_4_attain_extrapolated_std_kg_rcsoyhw =  np.einsum('c,rcsoyhw->rcsoyhw', convesion_factor, GAEZ_4_attain_extrapolated_std_rcsoyhw).astype(np.float16)


# Convert kg to tonnes
GAEZ_4_attain_extrapolated_mean_t_rcsoyhw = GAEZ_4_attain_extrapolated_mean_kg_rcsoyhw / 1000
GAEZ_4_attain_extrapolated_std_t_rcsoyhw = GAEZ_4_attain_extrapolated_std_kg_rcsoyhw / 1000

# Save the results
np.save('data/results/GAEZ_4_attain_extrapolated_mean_t_rcsoyhw.npy', GAEZ_4_attain_extrapolated_mean_t_rcsoyhw)
np.save('data/results/GAEZ_4_attain_extrapolated_std_t_rcsoyhw.npy', GAEZ_4_attain_extrapolated_std_t_rcsoyhw)




# Compare the yield with the yearbook data
if __name__ == '__main__':
    
    # Read the yearbook_yield
    yearbook_yield = pd.read_csv('data/results/yearbook_yield.csv')
    
    # Read the Province_mask_mean for computing the mean statistics for each province
    with rasterio.open('data/GAEZ_v4/Province_mask_mean.tif') as src:
        Province_mask_mean_phw = src.read()                                                                 # (province, h, w)


    # Compute the mean and std of attainable yield for each province
    GAEZ_4_attain_extrapolated_mean_t_rcsopy = np.einsum('rcsoyhw,phw->rcsopy', 
                                                            GAEZ_4_attain_extrapolated_mean_t_rcsoyhw, 
                                                            Province_mask_mean_phw)                         # (r, c, s, o, p, y)

    GAEZ_4_attain_extrapolated_std_t_rcsopy = np.einsum('rcsoyhw,phw->rcsopy',
                                                            GAEZ_4_attain_extrapolated_std_t_rcsoyhw,
                                                            Province_mask_mean_phw)                         # (r, c, s, o, p, y)

    # Filter the yield_array with specific rcp, c02_fertilization, and water_supply
    rcp = "RCP4.5" 
    c02_fertilization = "With CO2 Fertilization"
    
    GAEZ_yield_mean_df = ndarray_to_df(GAEZ_4_attain_extrapolated_mean_t_rcsopy, 'rcsopy', year_start=2010)
    GAEZ_yield_std_df = ndarray_to_df(GAEZ_4_attain_extrapolated_std_t_rcsopy, 'rcsopy', year_start=2010)
    
    GAEZ_yield_df = pd.merge(GAEZ_yield_mean_df, 
                             GAEZ_yield_std_df, 
                             on=['rcp', 'crop', 'water_supply', 'c02_fertilization', 'Province', 'attainable_year'], 
                             suffixes=('_mean', '_std'))
    
    GAEZ_yield_df = GAEZ_yield_df.query(f"rcp == '{rcp}' and c02_fertilization == '{c02_fertilization}'")
    GAEZ_yield_df['obs_ci_lower'] = GAEZ_yield_df['Value_mean'] - (GAEZ_yield_df['Value_std'] * 1.96)
    GAEZ_yield_df['obs_ci_upper'] = GAEZ_yield_df['Value_mean'] + (GAEZ_yield_df['Value_std'] * 1.96)


    # Plot the yield for each province of both yearbook and GAEZ
    plotnine.options.figure_size = (16, 6)
    plotnine.options.dpi = 100
    g = (
        plotnine.ggplot() +
        plotnine.geom_line(GAEZ_yield_df,
                        plotnine.aes(x='attainable_year', y='Value_mean', color='water_supply')
                        ) +
        plotnine.geom_ribbon(GAEZ_yield_df,
                            plotnine.aes(x='attainable_year', ymin='obs_ci_lower', ymax='obs_ci_upper', fill='water_supply'), alpha=0.5
                            ) +
        plotnine.geom_point(yearbook_yield, plotnine.aes(x='year', y='Yield (tonnes)')) +
        plotnine.facet_grid('crop~Province', scales='free_y') +
        plotnine.theme(axis_text_x=plotnine.element_text(rotation=45, hjust=1))
    )


