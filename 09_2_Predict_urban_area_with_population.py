import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import plotnine
import statsmodels.api as sm

from helper_func.parameters import UNIQUE_VALUES


# Read hist data
pop_total_and_ratio = pd.read_csv('data/results/step_9_2_total_pop_and_urban_ratio.csv')
urban_hist_area_km2 = pd.read_csv('data/results/step_9_1_urban_area_ext.csv')

# Read ssp future data
population_ssp = pd.read_csv('data/results/POP_NCP_pred.csv')
population_ssp['Population (million)'] = population_ssp['Value'] / 100

# Combine the data
data_hist_normal = pd.merge(
    pop_total_and_ratio,
    urban_hist_area_km2,
    on=['Province', 'year']
)

data_hist_normal['Province_idx'] = data_hist_normal['Province']\
    .map(lambda x: {v:k for k,v in dict(enumerate(UNIQUE_VALUES['Province'])).items()}[x])
    

# Use the total population and urban population ratio to get slopes and intercepts as priors
data_hist_sum = data_hist_normal.groupby('year')[['Population (million)', 'Area_cumsum_km2']].sum().reset_index()




df_anhui = data_hist_normal.query('Province == "Anhui" and year > 2010')

def linnear_ols(df):
    X = df['Population (million)']
    y = df['Area_cumsum_km2']
    X = sm.add_constant(X)
    model = sm.OLS(y, X)
    results = model.fit()
    return results.params

params = linnear_ols(df_anhui) 
a,b = int(params.const), int(params['Population (million)'])

with pm.Model() as m_anhui:
    # Priors
    sigma = pm.HalfNormal('sigma', sigma=100)
    intercept = pm.Normal('intercept', mu=a, sigma=100)
    slope_pop = pm.Normal('slope_pop', mu=b, sigma=100)
    
    # Data
    pop = pm.Data('pop', df_anhui['Population (million)'])
    area = pm.Data('area', df_anhui['Area_cumsum_km2'])
    
    # Linear model
    mu = pm.Deterministic('mu', intercept + slope_pop * pop)
    
    # Likelihood
    pm.StudentT('y', nu=9, mu=mu, sigma=sigma, observed=area)
    
    # Sample
    trace_anhui = pm.sample(2000, cores=1, chains=4)

# Get the posterior predictive
with m_anhui:
    ppc = pm.sample_posterior_predictive(trace_anhui)
    
ppc_df = ppc['posterior_predictive'].to_dataframe().reset_index()
ppc_df[['Province', 'year']] = df_anhui[['Province', 'year']].values.tolist() * ppc_df['chain'].nunique() * ppc_df['draw'].nunique()
ppc_df['year'] = ppc_df['year'].astype(int)

ppc_df_stats = ppc_df.groupby(['Province', 'year'])['y'].describe(percentiles=[0.025, 0.975]).reset_index()

g = (plotnine.ggplot()
        + plotnine.geom_line(
            df_anhui,
            plotnine.aes(x='year', y='Area_cumsum_km2', color='Province'))
         + plotnine.geom_ribbon(
             ppc_df_stats,
             plotnine.aes(x='year', ymin='2.5%', ymax='97.5%'),
             fill='grey',
             alpha=0.3)
        + plotnine.facet_wrap('~Province')
        + plotnine.theme_bw()
    )



# Build mc model       
with pm.Model(coords={'Province':UNIQUE_VALUES['Province']}) as linear_model:
    # Priors
    sigma = pm.HalfNormal('sigma', sigma=100, dims='Province')
    intercepts = pm.Normal('intercepts', mu=10000, sigma=5000, dims='Province')
    slopes_pop = pm.Normal('slopes_pop', mu=6000, sigma=1000, dims='Province')
    
    # Data
    pop = pm.Data('pop', data_hist_normal['Population (million)'], dims='obs_id')
    area = pm.Data('area', data_hist_normal['Area_cumsum_km2'], dims='obs_id')
    province = pm.Data('province', data_hist_normal['Province_idx'], dims='obs_id')
    
    # Linear model
    mu = pm.Deterministic(
        'mu', 
        intercepts[province] + slopes_pop[province] * pop,
        dims='obs_id')
    
    # Likelihood
    pm.Normal(
        'y', 
        mu=mu, 
        sigma=sigma[province], 
        observed=area, 
        dims='obs_id')
    
    # Sample
    trace = pm.sample(2000, cores=1, chains=4)
    
    
# Plot trace
az.plot_trace(trace, filter_vars="regex", var_names=["~mu"])



# Inference
with linear_model:
    ppc = pm.sample_posterior_predictive(trace)
    
ppc_df = ppc['posterior_predictive'].to_dataframe().reset_index()
ppc_df[['Province', 'year']] = data_hist_normal[['Province', 'year']].values.tolist() * ppc_df['chain'].nunique() * ppc_df['draw'].nunique()
ppc_df['year'] = ppc_df['year'].astype(int)

ppc_df_stats = ppc_df.groupby(['Province', 'year'])['y'].describe(percentiles=[0.34, 0.68]).reset_index()



    
    
# Sanity check

if __name__ == '__main__':
    
    # Plot total_Population vs total_Area
    plotnine.options.figure_size = (8, 6)
    plotnine.options.dpi = 100

    g = (plotnine.ggplot(data_hist_sum)
            + plotnine.geom_point(
                plotnine.aes(x='Population (million)', y='Area_cumsum_km2'))
            + plotnine.theme_bw()
        )
    g.save('data/results/fig_step_9_4_total_population_vs_total_area.svg')
    
    
    
    # Plot the population vs urban area for each province
    
    g = (plotnine.ggplot(data_hist_normal.query('year > 2010'))
         + plotnine.geom_point(
             plotnine.aes(x='year', y='Population (million)', color='Province'))
         + plotnine.facet_wrap('~Province', scales='free')
         + plotnine.theme_bw()
         )
    
    g = (plotnine.ggplot(data_hist_normal.query('year > 2010'))
         + plotnine.geom_point(
             plotnine.aes(x='year', y='urban_pop_ratio', color='Province'))
         + plotnine.facet_wrap('~Province', scales='free')
         + plotnine.theme_bw()
         )
    
    g = (plotnine.ggplot(data_hist_normal.query('year > 2010'))
         + plotnine.geom_point(
             plotnine.aes(x='year', y='Area_cumsum_km2', color='Province'))
         + plotnine.facet_wrap('~Province', scales='free')
         + plotnine.theme_bw()
         )
    
    
    g = (plotnine.ggplot(data_hist_normal.query('year > 2010'))
         + plotnine.geom_point(
             plotnine.aes(x='Population (million)', y='Area_cumsum_km2', color='Province'))
         + plotnine.facet_wrap('~Province', scales='free')
         + plotnine.theme_bw()
         )
    
    
    g.save('data/results/fig_step_9_5_population_vs_urban_area.svg')
    
    
    
    # Plot the posterior predictive
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