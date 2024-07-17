import math
import pandas as pd
import plotnine


# from helper_func import fit_linear_model
from helper_func import fit_linear_model
from helper_func.get_yearbook_records import get_yearbook_yield
from helper_func.parameters import BASE_YR, UNIQUE_VALUES

# Read the yearbook yield data
yearbook_yield = get_yearbook_yield().query('year >= 1990')
yearbook_yield = yearbook_yield.sort_values(['Province', 'crop', 'year'])
yearbook_yield_grouped = yearbook_yield.groupby(['Province', 'crop'])



# Function to fit a linear model to the data
yearbook_yield_fitted = pd.DataFrame()
for (province,crop), df in yearbook_yield_grouped:
    fitted_df = fit_linear_model(df)
    fitted_df.insert(0, 'Province', province)
    fitted_df.insert(1, 'crop', crop)
    fitted_df['obs_ci_lower'] = fitted_df['mean'] - (fitted_df['std'] / math.sqrt(len(fitted_df)) * 1.96)
    fitted_df['obs_ci_upper'] = fitted_df['mean'] + (fitted_df['std'] / math.sqrt(len(fitted_df)) * 1.96)
    
    yearbook_yield_fitted = pd.concat([yearbook_yield_fitted, fitted_df])
    
    
# Make the pred and yearbook the same at BASE_YR
yearbook_BASE = yearbook_yield.query(f'year == {BASE_YR}').copy()
pred_BASE = yearbook_yield_fitted.query(f'year == {BASE_YR}').copy()

diff = pred_BASE.merge(yearbook_BASE, on=['Province', 'crop'], suffixes=('_pred', '_yearbook'))
diff['diff'] = diff['Yield (tonnes)'] - diff['mean']
diff = diff[['Province', 'crop', 'diff']]

yearbook_yield_fitted = yearbook_yield_fitted.merge(diff, on=['Province', 'crop'], how='left')
yearbook_yield_fitted[['mean', 'std', 'obs_ci_lower', 'obs_ci_upper']] = yearbook_yield_fitted[['mean', 'std', 'obs_ci_lower', 'obs_ci_upper']] + yearbook_yield_fitted['diff'].values.reshape(-1, 1)


# If the prediction shows decrease in yield, set it to be the same as the <BASE_YR> of the yearbook data
for province in UNIQUE_VALUES['Province']:
    for crop in UNIQUE_VALUES['crop']:
        
        pred_df = yearbook_yield_fitted.query('Province == @province and crop == @crop')
        pred_mean = pred_df['mean'].mean()
        yearbook_base = yearbook_yield.query(f'Province == @province and crop == @crop and year == @BASE_YR')['Yield (tonnes)'].values[0]
        
        if pred_mean < yearbook_base:
            yearbook_yield_fitted.loc[(yearbook_yield_fitted['Province'] == province) & (yearbook_yield_fitted['crop'] == crop), 'mean'] = yearbook_base
            yearbook_yield_fitted.loc[(yearbook_yield_fitted['Province'] == province) & (yearbook_yield_fitted['crop'] == crop), 'std'] = pred_df['std']
            yearbook_yield_fitted.loc[(yearbook_yield_fitted['Province'] == province) & (yearbook_yield_fitted['crop'] == crop), 'obs_ci_lower'] = yearbook_base - (pred_df['std'] / math.sqrt(len(pred_df)) * 1.96)
            yearbook_yield_fitted.loc[(yearbook_yield_fitted['Province'] == province) & (yearbook_yield_fitted['crop'] == crop), 'obs_ci_upper'] = yearbook_base + (pred_df['std'] / math.sqrt(len(pred_df)) * 1.96)



# Convert dataframes to xarray
yearbook_yield_extrapolated = yearbook_yield_fitted.set_index(['Province', 'crop', 'year'])[['mean','std']].to_xarray()
yearbook_yield_extrapolated.to_netcdf('data/results/step_6_yearbook_yield_extrapolated.nc')




# Sanity check
if __name__ == '__main__':

    # Plot the data
    plotnine.options.figure_size = (16, 6)
    plotnine.options.dpi = 100
    
    g = (plotnine.ggplot() +
         plotnine.geom_point(yearbook_yield, plotnine.aes(x='year', y='Yield (tonnes)'), alpha=0.2, size=0.2) +
         plotnine.geom_line(yearbook_yield_fitted,
                            plotnine.aes(x='year', y='mean' )
                            ) +
         plotnine.geom_ribbon(yearbook_yield_fitted, 
                              plotnine.aes(x='year', ymin='obs_ci_lower', ymax='obs_ci_upper'), 
                              alpha=0.3
                              ) +
         plotnine.facet_grid('crop~Province') +
         plotnine.theme_bw() + 
         plotnine.labs(x='Year', y='Yield (t/ha)')
         )
    
    g.save('data/results/fig_step_6_yearbook_yield_extrapolated_t_ha.svg')


