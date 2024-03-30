import numpy as np
import pandas as pd
import plotnine
import statsmodels.api as sm


# from helper_func import fit_linear_model
from helper_func import fit_linear_model
from helper_func.parameters import BASE_YR, TARGET_YR, UNIQUE_VALUES


# Read the yearbook yield data
yearbook_yield = pd.read_csv('data/results/yearbook_yield.csv')

# Only consider the data from 1990 onwards
yearbook_yield = yearbook_yield.query('year >= 1990')


# Group the data by province
yearbook_yield_grouped = yearbook_yield.groupby(['Province', 'crop'])




# Function to fit a linear model to the data
fitted_dfs = []
for (province,crop), df in yearbook_yield_grouped:
    fitted_df = fit_linear_model(df)
    fitted_df.insert(0, 'Province', province)
    fitted_df.insert(1, 'crop', crop)
    fitted_dfs.append(fitted_df)
    
yearbook_yield_fitted = pd.concat(fitted_dfs)





# If the prediction shows decrease in yield, set it to be the same as the <BASE_YR> of the yearbook data
for province in UNIQUE_VALUES['Province']:
    for crop in UNIQUE_VALUES['crop']:
        
        pred_df = yearbook_yield_fitted.query('Province == @province and crop == @crop')
        pred_mean = pred_df['mean'].mean()
        yearbook_base = yearbook_yield.query('Province == @province and crop == @crop and year == @BASE_YR')['Yield (tonnes)'].values[0]
        
        if pred_mean < yearbook_base:
            yearbook_yield_fitted.loc[(yearbook_yield_fitted['Province'] == province) & (yearbook_yield_fitted['crop'] == crop), 'mean'] = yearbook_base
            yearbook_yield_fitted.loc[(yearbook_yield_fitted['Province'] == province) & (yearbook_yield_fitted['crop'] == crop), 'obs_ci_lower'] = yearbook_base - (pred_df['std'] * 1.96)
            yearbook_yield_fitted.loc[(yearbook_yield_fitted['Province'] == province) & (yearbook_yield_fitted['crop'] == crop), 'obs_ci_upper'] = yearbook_base + (pred_df['std'] * 1.96)


# Save the data
yearbook_yield_fitted = yearbook_yield_fitted.sort_values(['Province', 'crop', 'year'])
yearbook_yield_mean_fitted_pcy = yearbook_yield_fitted['mean'].values.reshape(len(UNIQUE_VALUES['Province']), 
                                                                       len(UNIQUE_VALUES['crop']),
                                                                       len(UNIQUE_VALUES['simulation_year']))

yearbook_yield_std_fitted_pcy = yearbook_yield_fitted['std'].values.reshape(len(UNIQUE_VALUES['Province']), 
                                                                     len(UNIQUE_VALUES['crop']),
                                                                     len(UNIQUE_VALUES['simulation_year']))


np.save('data/results/yearbook_yield_mean_fitted_pcy.npy', yearbook_yield_mean_fitted_pcy)
np.save('data/results/yearbook_yield_std_fitted_pcy.npy', yearbook_yield_std_fitted_pcy)




# Sanity check
if __name__ == '__main__':

    plotnine.options.figure_size = (16, 6)
    plotnine.options.dpi = 100
    
    g = (plotnine.ggplot() +
         plotnine.geom_point(yearbook_yield, plotnine.aes(x='year', y='Yield (tonnes)'),alpha=0.5,size=0.2) +
         plotnine.geom_line(yearbook_yield_fitted, plotnine.aes(x='year', y='mean')) +
         plotnine.geom_ribbon(yearbook_yield_fitted, 
                              plotnine.aes(x='year', ymin='obs_ci_lower', ymax='obs_ci_upper'), alpha=0.5,fill='grey') +
         plotnine.facet_grid('crop~Province') +
         plotnine.theme_bw()
         )


