import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import plotnine

from sklearn.preprocessing import StandardScaler
from helper_func.parameters import UNIQUE_VALUES


# Define the ceiling of the urban population ratio (%)
cellings = {
    'Anhui': 75,
    'Beijing': 90,
    'Hebei': 75,
    'Henan': 75,
    'Jiangsu': 80,
    'Shandong': 75,
    'Tianjin': 85,
}



# Read hist data
pop_total_and_ratio = pd.read_csv('data/results/step_9_1_2_total_pop_and_urban_ratio.csv')
pop_total_and_ratio = pop_total_and_ratio.query('year > 2010').copy()
pop_total_and_ratio = pop_total_and_ratio[['Province', 'year', 'Population (million)', 'urban_pop_ratio']]
pop_total_and_ratio = pop_total_and_ratio.sort_values(['Province', 'year']).reset_index(drop=True)
pop_total_and_ratio['Province_idx'] = pop_total_and_ratio['Province']\
    .map(lambda x: {v:k for k,v in dict(enumerate(UNIQUE_VALUES['Province'])).items()}[x])

pop_total_and_ratio['ceiling'] = pop_total_and_ratio['Province'].map(cellings)



# Standardize the data
def fit_scaler(x):
    scaler = StandardScaler()
    return scaler.fit(x.values.reshape(-1, 1))

scalers_yr = pop_total_and_ratio.groupby('Province')['year']\
            .agg(scaler=lambda x: fit_scaler(x))\
            .to_dict()['scaler']

pop_total_and_ratio['year_normalized'] = pop_total_and_ratio\
    .groupby('Province')['year']\
    .transform(lambda x: scalers_yr[x.name].transform(x.values.reshape(-1, 1)).flatten())



####################################################################
# Logistic regression to predict the urban population ratio
####################################################################

with pm.Model(coords={'Province':UNIQUE_VALUES['Province']}) as logistic_model:
    # Priors for unknown model parameters
    intercept = pm.Normal('intercept', mu=0, sigma=10, dims='Province')
    slope = pm.Normal('slope', mu=0, sigma=10, dims='Province')
    sigma = pm.HalfCauchy('sigma', beta=0.5, dims='Province')
    
    # Data
    pop_ratio = pm.Data('pop_ratio', pop_total_and_ratio['urban_pop_ratio'], dims='obs')
    yr_normalized = pm.Data('yr_normalized', pop_total_and_ratio['year_normalized'], dims='obs')
    province = pm.Data('province', pop_total_and_ratio['Province_idx'], dims='obs')
    ceiling = pm.Data('ceiling', pop_total_and_ratio['ceiling'], dims='obs')
    
    # Logistic regression
    p = pm.Deterministic(
        'p', 
        ceiling / (1 + np.exp(-(intercept[province] + slope[province] * yr_normalized)) ),
        dims='obs')
    
    # Likelihood (sampling distribution) of observations
    y_obs = pm.Normal(
        'y_obs', 
        mu=p, 
        sigma=sigma[province], 
        observed=pop_ratio, 
        dims='obs')
    
    # Sample from the posterior
    trace = pm.sample(1000, cores=1, chains=4)



# ---------------------------- Inference --------------------------------

# Check the trace parameters
az.plot_trace(trace, filter_vars="regex", var_names=["~^p"])

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
    
future_data['ceiling'] = future_data['Province'].map(cellings)

# Update the data to predict the urban population ratio
with logistic_model:
    # Update the data in the model
    pm.set_data({
        'yr_normalized': future_data['year_normalized'],
        'province': future_data['Province_idx'],
        'ceiling': future_data['ceiling'],
        'pop_ratio': np.zeros(len(future_data))
    })
    ppc = pm.sample_posterior_predictive(trace)
    

# Extract the predicted urban population ratio
ppc_df = ppc['posterior_predictive'].to_dataframe().reset_index()
ppc_df[['Province', 'year']] = future_data[['Province', 'year']].values.tolist() * ppc_df['chain'].nunique() * ppc_df['draw'].nunique()
ppc_df['year'] = ppc_df['year'].astype(int)

ppc_df_stats = ppc_df\
    .groupby(['Province', 'year'])['y_obs']\
    .describe(percentiles=[0.025, 0.975]).reset_index()

# Save to disk
ppc_df_stats.to_csv('data/results/step_9_2_predict_urban_pop_ratio.csv', index=False)

    
# Sanity check
if __name__ == '__main__':
    
    plotnine.options.figure_size = (8, 6)
    plotnine.options.dpi = 100
    
    # Plot the predicted urban population ratio
    g = (plotnine.ggplot()
        + plotnine.geom_line(
            pop_total_and_ratio,
            plotnine.aes(x='year', y='urban_pop_ratio', color='Province'))
         + plotnine.geom_ribbon(
             ppc_df_stats,
             plotnine.aes(x='year', ymin='2.5%', ymax='97.5%'),
             fill='grey',
             alpha=0.3)
        + plotnine.facet_wrap('~Province', scales='free')
        + plotnine.theme_bw()
    )
    
    g.save('data/results/fig_step_9_2_predict_urban_pop_ratio.svg')
    