import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import plotnine
from pyparsing import col
import statsmodels.api as sm

from helper_func.parameters import UNIQUE_VALUES
from sklearn.preprocessing import StandardScaler

# Read hist data
urban_hist_area_km2 = pd.read_csv('data/results/step_9_1_1_urban_area_ext.csv')
population_hist = pd.read_csv('data/results/step_9_1_2_total_pop_and_urban_ratio.csv')
population_area_hist = population_hist.merge(urban_hist_area_km2, on=['Province', 'year'])
population_area_hist['Province_idx'] = population_area_hist['Province']\
    .map(lambda x: {v:k for k,v in dict(enumerate(UNIQUE_VALUES['Province'])).items()}[x])

# Read future data
popupation_ratio_future = pd.read_csv('data/results/step_9_2_predict_urban_pop_ratio.csv')
population_total_future = pd.read_csv('data/results/POP_NCP_pred.csv')
population_total_future['Population (million)'] = population_total_future['Value'] / 100


# Standardize the data
def fit_scaler(x):
    scaler = StandardScaler()
    return scaler.fit(x.values.reshape(-1, 1))

scalers_yr = population_area_hist.groupby('Province')['year']\
            .agg(scaler=lambda x: fit_scaler(x))\
            .to_dict()['scaler']

scalers_area = population_area_hist.groupby('Province')['Area_cumsum_km2']\
            .agg(scaler=lambda x: fit_scaler(x))\
            .to_dict()['scaler']
            
scalers_pop = population_area_hist.groupby('Province')['Population (million)']\
            .agg(scaler=lambda x: fit_scaler(x))\
            .to_dict()['scaler']
            
scalers_ratio = population_area_hist.groupby('Province')['urban_pop_ratio']\
            .agg(scaler=lambda x: fit_scaler(x))\
            .to_dict()['scaler']

population_area_hist['year_normalized'] = population_area_hist\
    .groupby('Province')['year']\
    .transform(lambda x: scalers_yr[x.name].transform(x.values.reshape(-1, 1)).flatten())
    
population_area_hist['area_normalized'] = population_area_hist\
    .groupby('Province')['Area_cumsum_km2']\
    .transform(lambda x: scalers_area[x.name].transform(x.values.reshape(-1, 1)).flatten())
    
population_area_hist['pop_normalized'] = population_area_hist\
    .groupby('Province')['Population (million)']\
    .transform(lambda x: scalers_pop[x.name].transform(x.values.reshape(-1, 1)).flatten())
    
population_area_hist['ratio_normalized'] = population_area_hist\
    .groupby('Province')['urban_pop_ratio']\
    .transform(lambda x: scalers_ratio[x.name].transform(x.values.reshape(-1, 1)).flatten())



#####################################################################
# Linear regression to predict the urban area
#####################################################################

with pm.Model(coords={'Province':UNIQUE_VALUES['Province']}) as linear_model:
    # Priors for unknown model parameters
    intercept_pop = pm.Normal('intercept_pop', mu=0, sigma=3, dims='Province')
    intercept_ratio = pm.Normal('intercept_ratio', mu=0, sigma=3, dims='Province')
    slope_pop = pm.Normal('slope_pop', mu=0, sigma=3, dims='Province')
    slope_ratio = pm.Normal('slope_ratio', mu=0, sigma=3, dims='Province')
    sigma = pm.HalfCauchy('sigma', beta=0.5, dims='Province')
    
    # Data
    area_normalized = pm.Data('area_normalized', population_area_hist['area_normalized'], dims='obs')
    pop_normalized = pm.Data('pop_normalized', population_area_hist['pop_normalized'], dims='obs')
    ratio_normalized = pm.Data('ratio_normalized', population_area_hist['ratio_normalized'], dims='obs')
    province = pm.Data('province', population_area_hist['Province_idx'], dims='obs')
    
    # Linear regression model
    mu = pm.Deterministic('mu',
                          (intercept_pop[province] + slope_pop[province] * pop_normalized
                           + intercept_ratio[province] + slope_ratio[province] * ratio_normalized), 
                          dims='obs')
    
    # Likelihood       
    area = pm.Normal('area', 
                     mu=mu, 
                     sigma=sigma[province], 
                     observed=area_normalized, 
                     dims='obs')
    
    # Posterior
    linear_trace = pm.sample(2000, tune=2000, cores=1)



import matplotlib.pyplot as plt
az.plot_trace(linear_trace, filter_vars="regex", var_names=["~area"])
plt.show()

# ---------------------------- Inference --------------------------------

# Check the trace parameters
az.plot_trace(linear_trace, filter_vars="regex", var_names=["~^p"])

# Construct the future data
future_years = np.arange(2020, 2101)
province_indices = np.arange(len(UNIQUE_VALUES['Province']))
future_data = pd.DataFrame({
    'year': np.tile(future_years, len(province_indices)),
    'Province': np.repeat(UNIQUE_VALUES['Province'], len(future_years)),
    'Province_idx': np.repeat(province_indices, len(future_years))
})

future_data['year_normalized'] = future_data\
    .groupby('Province')['year']\
    .transform(lambda x: scalers_yr[x.name].transform(x.values.reshape(-1, 1)).flatten())
    

# Update the data to predict the urban population ratio
with linear_model:
    # Update the data in the model
    pm.set_data({
        'yr_normalized': future_data['year_normalized'],
        'province': future_data['Province_idx'],
        'ceiling': future_data['ceiling'],
        'pop_ratio': np.zeros(len(future_data))
    })
    
    
    
# Sanity check

if __name__ == '__main__':
    
    plotnine.options.figure_size = (8, 6)
    plotnine.options.dpi = 100
    
    
    # Plot the corelation between Population and Area
    g =(plotnine.ggplot(population_area_hist)
        + plotnine.geom_point(
            plotnine.aes(x='pop_normalized', y='area_normalized', color='Province'))
        + plotnine.facet_wrap('~Province', scales='free')
        + plotnine.geom_abline(intercept=0, slope=1, color='grey', linetype='dashed')
        + plotnine.theme_bw()
    )
    g.save('data/results/fig_step_9_3_1_population_vs_area.svg')
    
    g = (plotnine.ggplot(population_area_hist)
        + plotnine.geom_point(
            plotnine.aes(x='ratio_normalized', y='area_normalized', color='Province'))
        + plotnine.geom_abline(intercept=0, slope=1, color='grey', linetype='dashed')
        + plotnine.facet_wrap('~Province', scales='free')
        + plotnine.theme_bw()
    )
    g.save('data/results/fig_step_9_3_2_ratio_vs_area.svg')

    
    
    # Plot total_Population vs total_Area
    
    
    # Plot the predictive
    g = (plotnine.ggplot()
        + plotnine.geom_line(
            data_hist_normal,
            plotnine.aes(x='year', y='Area_cumsum_km2', color='Province'))
        # + plotnine.geom_line(
        #     ppc_df_stats,
        #     plotnine.aes(x='year', y='50%'))
         + plotnine.geom_ribbon(
             ppc_df_stats,
             plotnine.aes(x='year', ymin='34%', ymax='68%'),
             fill='grey',
             alpha=0.3)
        + plotnine.facet_wrap('~Province')
        + plotnine.theme_bw()
    )
    g.save('data/results/fig_step_9_6_posterior_predictive.svg')