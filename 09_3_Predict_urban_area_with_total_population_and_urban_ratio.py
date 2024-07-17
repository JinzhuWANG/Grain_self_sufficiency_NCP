import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import plotnine

from helper_func.parameters import UNIQUE_VALUES
from sklearn.preprocessing import StandardScaler

# Read hist data
urban_hist_area_km2 = pd.read_csv('data/results/step_9_1_1_urban_area_ext.csv')
population_hist = pd.read_csv('data/results/step_9_1_2_total_pop_and_urban_ratio.csv').query('year > 2010 and year < 2020')
population_area_hist = population_hist.merge(urban_hist_area_km2, on=['Province', 'year'])
population_area_hist['Province_idx'] = population_area_hist['Province']\
    .map(lambda x: {v:k for k,v in dict(enumerate(UNIQUE_VALUES['Province'])).items()}[x])

# Read future data
popupation_ratio_future = pd.read_csv('data/results/step_9_2_predict_urban_pop_ratio.csv')[['Province', 'year','mean', 'std']]
popupation_ratio_future['ratio_sample'] = popupation_ratio_future\
    .apply(lambda x: np.random.normal(x['mean'], x['std'], 100), axis=1)
        
population_total_future = pd.read_csv('data/results/POP_NCP_pred.csv')
population_total_future['Population (million)'] = population_total_future['Value'] / 100


# Standardize the data
def fit_scaler(x):
    scaler = StandardScaler()
    return scaler.fit(x.values.reshape(-1, 1))


scalers_area = population_area_hist.groupby('Province')['Area_cumsum_km2']\
            .agg(scaler=lambda x: fit_scaler(x))\
            .to_dict()['scaler']
            
scalers_pop = population_area_hist.groupby('Province')['Population (million)']\
            .agg(scaler=lambda x: fit_scaler(x))\
            .to_dict()['scaler']
            
scalers_ratio = population_area_hist.groupby('Province')['urban_pop_ratio']\
            .agg(scaler=lambda x: fit_scaler(x))\
            .to_dict()['scaler']


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
    # Hyperpriors
    intercept_mu = pm.Normal("intercept_mu", 0, sigma=1)
    intercept_sigma = pm.HalfNormal("intercept_sigma", sigma=2)
    slope_mu = pm.Normal("slope_mu", 0, sigma=1)
    slope_sigma = pm.HalfNormal("slope_sigma", sigma=2)
    sigma_hyperprior = pm.HalfNormal("sigma_hyperprior", sigma=0.5)
    
    
    # Priors for unknown model parameters
    sigma = pm.HalfNormal('sigma', sigma=sigma_hyperprior, dims='Province')
    
    intercept_offset = pm.Normal('intercept_offset', mu=0, sigma=1, dims='Province')
    intercept = pm.Deterministic('intercept', intercept_mu + intercept_sigma * intercept_offset, dims='Province')
    
    slope_pop_offset = pm.Normal('slope_pop_offset', mu=0, sigma=1, dims='Province')
    slope_pop = pm.Deterministic('slope_pop', slope_mu + slope_pop_offset * slope_sigma, dims='Province')
    
    slope_ratio_offset = pm.Normal('slope_ratio_offset', mu=0, sigma=1, dims='Province')
    slope_ratio = pm.Deterministic('slope_ratio', slope_mu + slope_ratio_offset * slope_sigma, dims='Province')
    
    
    # Data
    area_normalized = pm.Data('area_normalized', population_area_hist['area_normalized'], dims='obs')
    pop_normalized = pm.Data('pop_normalized', population_area_hist['pop_normalized'], dims='obs')
    ratio_normalized = pm.Data('ratio_normalized', population_area_hist['ratio_normalized'], dims='obs')
    province = pm.Data('province', population_area_hist['Province_idx'], dims='obs')
    
    # Linear regression model
    mu = pm.Deterministic('mu',
                          (intercept[province] 
                           + slope_pop[province] * pop_normalized
                           + slope_ratio[province] * ratio_normalized), 
                          dims='obs')
    
    # Likelihood       
    pm.StudentT('area', 
                nu=population_area_hist['year'].nunique(),
                mu=mu, 
                sigma=sigma[province], 
                observed=area_normalized, 
                dims='obs')

    # Posterior
    linear_trace = pm.sample(2000, tune=2000, cores=1)



# ---------------------------- Inference --------------------------------

# Check the trace parameters
az.plot_trace(linear_trace, filter_vars="regex", var_names=["~area"])


# Construct the future data
future_data = population_total_future.merge(popupation_ratio_future, on=['Province', 'year'])
future_data = future_data.explode('ratio_sample')
future_data['Province_idx'] = future_data['Province'].map(lambda x: {v:k for k,v in dict(enumerate(UNIQUE_VALUES['Province'])).items()}[x])
future_data['pop_normalized'] = future_data\
    .groupby('Province')['Population (million)']\
    .transform(lambda x: scalers_pop[x.name].transform(x.values.reshape(-1, 1)).flatten())
    
future_data['ratio_normalized'] = future_data\
    .groupby('Province')['ratio_sample']\
    .transform(lambda x: scalers_ratio[x.name].transform(x.values.reshape(-1, 1)).flatten())
    


# Update the data to predict the urban population ratio
with linear_model:
    # Update the data in the model
    pm.set_data({
        'province': future_data['Province_idx'],
        'pop_normalized': future_data['pop_normalized'],
        'ratio_normalized': future_data['ratio_normalized'],
        'area_normalized': np.zeros(len(future_data)) # Dummy data
    })
    
    # Posterior predictive
    linear_ppc = pm.sample_posterior_predictive(linear_trace)
    
    
    
# Extract the predicted urban population ratio
ppc_df = linear_ppc['posterior_predictive']
ppc_df['SSP'] = (('obs'), future_data['SSP'])
ppc_df['year'] = (('obs'), future_data['year'])
ppc_df['Province'] = (('obs'), future_data['Province'])

ppc_df = ppc_df.to_dataframe().reset_index()
ppc_df['area_km2'] = ppc_df\
    .groupby(['Province'])['area']\
    .transform(lambda x: scalers_area[x.name].inverse_transform(x.values.reshape(-1, 1)).flatten())

ppc_df_stats = ppc_df\
    .groupby(['Province', 'year','SSP'])['area_km2']\
    .describe(percentiles=[0.025, 0.975, 0.34, 0.68]).reset_index()



# Force the pred data agrees with the hist data in 2020
hist_area_2020 = urban_hist_area_km2.query('year == 2020').sort_values('Province')
pred_area_2020 = ppc_df_stats.query('year == 2020').groupby('Province')[['mean']].mean().reset_index().sort_values('Province')
diff = dict(zip(
    hist_area_2020['Province'].values,
    hist_area_2020['Area_cumsum_km2'].values - pred_area_2020['mean'].values))

ppc_df_stats_adj = ppc_df_stats.copy().sort_values(['Province','year'])
subtract_cols = [i for i in ppc_df_stats_adj.columns if not i in ['Province','year','SSP','count']]
ppc_df_stats_adj[subtract_cols] = ppc_df_stats_adj\
    .groupby('Province')[subtract_cols + ['Province']]\
    .apply(lambda x: x[subtract_cols] + diff[x['Province'].values[0]]).reset_index(drop=True)

ppc_df_stats_adj.to_csv('data/results/step_9_3_1_predict_urban_area_adj.csv', index=False)



# Stop urban area from decreasing
ppc_df_stats_adj_no_decrease = pd.DataFrame()

for idx,df in ppc_df_stats_adj.groupby( ['Province','SSP']):
    # Get the index urban area is the highest
    max_idx = df['mean'].idxmax()
    # Fill the values after the max_idx with the max value
    df.loc[max_idx+1:, subtract_cols] = df.loc[max_idx, subtract_cols].values
    
    ppc_df_stats_adj_no_decrease = pd.concat([ppc_df_stats_adj_no_decrease, df])

ppc_df_stats_adj_no_decrease.to_csv('data/results/step_9_3_2_predict_urban_area_adj_no_decrease.csv', index=False)




# Sanity check
if __name__ == '__main__':
    
    plotnine.options.figure_size = (10, 6)
    plotnine.options.dpi = 100
    
    ppc_df_stats_adj = pd.read_csv('data/results/step_9_3_1_predict_urban_area_adj.csv')
    ppc_df_stats_adj_no_decrease = pd.read_csv('data/results/step_9_3_2_predict_urban_area_adj_no_decrease.csv')
    
    
    # Plot the corelation between Population and Area
    g =(plotnine.ggplot(population_area_hist)
        + plotnine.geom_point(
            plotnine.aes(x='pop_normalized', y='area_normalized', color='Province'))
        + plotnine.facet_wrap('~Province', scales='free')
        + plotnine.geom_abline(intercept=0, slope=1, color='grey', linetype='dashed')
        + plotnine.theme_bw()
        + plotnine.labs(x='Population (normalized)', y='Urban Area (normalized)')
    )
    g.save('data/results/fig_step_9_3_1_population_vs_area.svg')
    
    # Plot total_Population vs total_Area
    g = (plotnine.ggplot(population_area_hist)
        + plotnine.geom_point(
            plotnine.aes(x='ratio_normalized', y='area_normalized', color='Province'))
        + plotnine.geom_abline(intercept=0, slope=1, color='grey', linetype='dashed')
        + plotnine.facet_wrap('~Province', scales='free')
        + plotnine.theme_bw()
        + plotnine.labs(x='Urban Population Ratio (normalized)', y='Urban Area (normalized)')
    )
    g.save('data/results/fig_step_9_3_2_ratio_vs_area.svg')

    
    # Plot the predictive
    g = (plotnine.ggplot()
        + plotnine.geom_line(
            urban_hist_area_km2,
            plotnine.aes(x='year', y='Area_cumsum_km2'))
         + plotnine.geom_ribbon(
             ppc_df_stats_adj,
             plotnine.aes(x='year', ymin='2.5%', ymax='97.5%', fill='SSP'),
             alpha=0.3)
        + plotnine.facet_wrap('~Province', scales='free')
        + plotnine.theme_bw()
    )
    g.save('data/results/fig_step_9_3_3_future_urban_area_km2_1std_mixed_effects.svg')
    
    
    g = (plotnine.ggplot()
        + plotnine.geom_line(
            urban_hist_area_km2,
            plotnine.aes(x='year', y='Area_cumsum_km2'))
         + plotnine.geom_ribbon(
             ppc_df_stats_adj_no_decrease,
             plotnine.aes(x='year', ymin='2.5%', ymax='97.5%', fill='SSP'),
             alpha=0.3)
        + plotnine.facet_wrap('~Province', scales='free')
        + plotnine.theme_bw()
    )
    g.save('data/results/fig_step_9_3_4_future_urban_area_km2_1std_mixed_effects_no_decrease.svg')