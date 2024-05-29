import pandas as pd
import plotnine


# from helper_func import fit_linear_model
from helper_func import fit_linear_model
from helper_func.get_yearbook_records import get_yearbook_yield
from helper_func.parameters import BASE_YR, UNIQUE_VALUES

# Read the yearbook yield data
yearbook_yield = get_yearbook_yield().query('year >= 1990')
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
        yearbook_base = yearbook_yield.query(f'Province == @province and crop == @crop and year == @BASE_YR')['Yield (tonnes)'].values[0]
        
        if pred_mean < yearbook_base:
            yearbook_yield_fitted.loc[(yearbook_yield_fitted['Province'] == province) & (yearbook_yield_fitted['crop'] == crop), 'mean'] = yearbook_base
            yearbook_yield_fitted.loc[(yearbook_yield_fitted['Province'] == province) & (yearbook_yield_fitted['crop'] == crop), 'obs_ci_lower'] = yearbook_base - (pred_df['std'] * 1.96)
            yearbook_yield_fitted.loc[(yearbook_yield_fitted['Province'] == province) & (yearbook_yield_fitted['crop'] == crop), 'obs_ci_upper'] = yearbook_base + (pred_df['std'] * 1.96)


# Convert dataframes to xarray
yearbook_yield_extrapolated = yearbook_yield_fitted.set_index(['Province', 'crop', 'year'])[['mean','std']].to_xarray()
yearbook_yield_extrapolated.to_netcdf('data/results/step_6_yearbook_yield_extrapolated.nc')




# Sanity check
if __name__ == '__main__':

    # Plot the data
    plotnine.options.figure_size = (16, 6)
    plotnine.options.dpi = 100
    
    g = (plotnine.ggplot() +
         plotnine.geom_point(yearbook_yield, plotnine.aes(x='year', y='Yield (tonnes)'), alpha=0.5, size=0.2) +
         plotnine.geom_line(yearbook_yield_fitted,
                            plotnine.aes(x='year', y='mean' )
                            ) +
         plotnine.geom_ribbon(yearbook_yield_fitted, 
                              plotnine.aes(x='year', ymin='obs_ci_lower', ymax='obs_ci_upper'), 
                              alpha=0.5
                              ) +
         plotnine.facet_grid('crop~Province') +
         plotnine.theme_bw()
         )


