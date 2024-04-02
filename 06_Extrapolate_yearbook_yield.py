import numpy as np
import pandas as pd
import plotnine


# from helper_func import fit_linear_model
from helper_func import fit_linear_model, ndarray_to_df
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


# Convert dataframes to numpy arrays
yearbook_yield_fitted = yearbook_yield_fitted.sort_values(['Province', 'crop', 'year'])
yearbook_yield_mean_fitted_pcy = yearbook_yield_fitted['mean'].values.reshape(len(UNIQUE_VALUES['Province']), 
                                                                       len(UNIQUE_VALUES['crop']),
                                                                       len(UNIQUE_VALUES['simulation_year']))      # (p, c, y) 

yearbook_yield_std_fitted_pcy = yearbook_yield_fitted['std'].values.reshape(len(UNIQUE_VALUES['Province']), 
                                                                     len(UNIQUE_VALUES['crop']),
                                                                     len(UNIQUE_VALUES['simulation_year']))        # (p, c, y)


# Divide the yield by its <BASE_YR> value, so we get the relative scale change in yield
yearbook_yield_mean_fitted_scale_pcy = yearbook_yield_mean_fitted_pcy / yearbook_yield_mean_fitted_pcy[:, :, 0:1]   # (p, c, y)
yearbook_yield_std_fitted_scale_pcy = yearbook_yield_std_fitted_pcy / yearbook_yield_mean_fitted_pcy[:, :, 0:1]     # (p, c, y)  


# Add an extra dimension to indicate the water supply
yearbook_yield_mean_fitted_scale_pcsy = yearbook_yield_mean_fitted_scale_pcy[:, :, None, :].repeat(len(UNIQUE_VALUES['water_supply']), axis=2)  # (p, c, s, y)
yearbook_yield_std_fitted_scale_pcsy = yearbook_yield_std_fitted_scale_pcy[:, :, None, :].repeat(len(UNIQUE_VALUES['water_supply']), axis=2)    # (p, c, s, y)


# The std of Wetland Rice for dryland is 0, so we set it to 0
yearbook_yield_std_fitted_scale_pcsy[:, 
                                    UNIQUE_VALUES['crop'].index('Wetland rice'),
                                    UNIQUE_VALUES['water_supply'].index('Dryland'), 
                                    :] = 0


# Save the data
np.save('data/results/yearbook_yield_mean_fitted_scale_pcsy.npy', yearbook_yield_mean_fitted_scale_pcsy.astype(np.float16))
np.save('data/results/yearbook_yield_std_fitted_scale_pcsy.npy', yearbook_yield_std_fitted_scale_pcsy.astype(np.float16))



# Sanity check
if __name__ == '__main__':
    
    # Get the dataframes for the yearbook yield
    yearbook_yield_mean_fitted_scale_pcsy_df = ndarray_to_df(yearbook_yield_mean_fitted_scale_pcsy, 'pcsy')
    yearbook_yield_std_fitted_pcsy_df = ndarray_to_df(yearbook_yield_std_fitted_scale_pcsy, 'pcsy')
    
    # Merge the dataframes
    yearbook_yield_fitted_df = pd.merge(yearbook_yield_mean_fitted_scale_pcsy_df, 
                                        yearbook_yield_std_fitted_pcsy_df, 
                                        on=['Province', 'crop', 'water_supply', 'simulation_year'],
                                        suffixes=('_mean_scale', '_std_scale'))
    
    # Multiply the yiled of <BASE_YR> with the relative scale change to get the yield for each year
    yearbook_yield_fitted_df['mean'] = yearbook_yield_fitted_df.apply(lambda x: x['Value_mean_scale'] * yearbook_yield_fitted.query('Province == @x.Province and crop == @x.crop and year == @BASE_YR')['mean'].values[0], axis=1)
    yearbook_yield_fitted_df['Value_std'] = yearbook_yield_fitted_df.apply(lambda x: x['Value_mean_scale'] * yearbook_yield_fitted.query('Province == @x.Province and crop == @x.crop and year == @BASE_YR')['std'].values[0], axis=1)
    yearbook_yield_fitted_df['obs_ci_lower'] = yearbook_yield_fitted_df['mean'] - 1.96 * yearbook_yield_fitted_df['Value_std']
    yearbook_yield_fitted_df['obs_ci_upper'] = yearbook_yield_fitted_df['mean'] + 1.96 * yearbook_yield_fitted_df['Value_std']

    # Plot the data
    plotnine.options.figure_size = (16, 6)
    plotnine.options.dpi = 100
    
    g = (plotnine.ggplot() +
         plotnine.geom_point(yearbook_yield, plotnine.aes(x='year', y='Yield (tonnes)'),alpha=0.5,size=0.2) +
         plotnine.geom_line(yearbook_yield_fitted_df,
                            plotnine.aes(x='simulation_year', y='mean', color='water_supply', )
                            ) +
         plotnine.geom_ribbon(yearbook_yield_fitted_df, 
                              plotnine.aes(x='simulation_year', ymin='obs_ci_lower', ymax='obs_ci_upper', fill='water_supply'), 
                              alpha=0.5
                              ) +
         plotnine.facet_grid('crop~Province') +
         plotnine.theme_bw()
         )


