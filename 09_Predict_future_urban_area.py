import pandas as pd
import pymc as pm
import plotnine
import arviz as az

from helper_func.parameters import UNIQUE_VALUES


# Read the historical data
urban_area_hist = pd.read_csv('data/results/area_hist_df.csv')          # unit: km2

GDP_NCP_hist = pd.read_csv('data/results/GDP_NCP.csv')                  # unit: (PPP) billion US$2005/yr                                        
POP_NCP_hist = pd.read_csv('data/results/POP_NCP.csv')                  # unit: 10k person                          


# Read the pred GDP and population data
GDP_NCP_pred = pd.read_csv('data/results/GDP_NCP_pred.csv')             # unit: (PPP) billion US$2005/yr
POP_NCP_pred = pd.read_csv('data/results/POP_NCP_pred.csv')             # unit: 10k person
GDP_POP_NCP_pred = GDP_NCP_pred.merge(POP_NCP_pred, on=['year','Province','SSP'])


# Modeling the urban_area ~ GDP + Population
area_GDP_POP = urban_area_hist.merge(GDP_NCP_hist, on=['year','Province']).merge(POP_NCP_hist, on=['year','Province'])
area_GDP_POP['Province_code'] = area_GDP_POP['Province'].map({v:k for k,v in enumerate(UNIQUE_VALUES['Province'])})



if __name__ == '__main__':
    with pm.Model(coords={'group':UNIQUE_VALUES['Province']}) as mix_model:
        
        # Overall effects
        intercept_all_mu = pm.Normal('intercept_all', mu=0, sigma=100)
        intercept_all_sigma = pm.HalfNormal('intercept_all_sigma', sigma=10)
        slope_all_mu = pm.Normal('slope_all_mu', mu=0, sigma=10)
        slope_all_sigma = pm.HalfNormal('slope_all_sigma', sigma=10)
        sigma_all = pm.HalfNormal('sigma_all', sigma=10)
        
        
        # Priors
        sigma_group = pm.HalfNormal('sigma_group', sigma=sigma_all, dims='group')
        
        intercept_effect = pm.Normal('intercept_effect', mu=0, sigma=10, dims='group')
        intercept_group = pm.Deterministic('intercept_group', intercept_all_mu + intercept_all_sigma * intercept_effect, dims='group')
        
        slope_GDP_effect = pm.Normal('slope_GDP_effect', mu=0, sigma=10, dims='group')
        slope_GDP_group = pm.Deterministic('slope_GDP_group', slope_all_mu + slope_all_sigma * slope_GDP_effect, dims='group')
        slope_POP_effect = pm.Normal('slope_POP_effect', mu=0, sigma=10, dims='group')
        slope_POP_group = pm.Deterministic('slope_POP_group', slope_all_mu + slope_all_sigma * slope_POP_effect, dims='group')
        
        # Observed data
        GDP = pm.MutableData('GDP', area_GDP_POP['GDP2SSP'], dims='obs_id')
        POP = pm.MutableData('POP', area_GDP_POP['Value'], dims='obs_id')
        group_idx = pm.MutableData('group_idx', area_GDP_POP['Province_code'], dims='obs_id')
        
        # Deterministic
        mu = pm.Deterministic('mu', intercept_group[group_idx] + slope_GDP_group[group_idx] * GDP + slope_POP_group[group_idx] * POP)
        
        # Likelihood
        pm.Normal('y', mu=mu, sigma=sigma_group[group_idx], observed=area_GDP_POP['Area_cumsum_km2'], dims='obs_id')
        
        # Sample
        step = pm.NUTS(target_accept=0.95, max_treedepth=15)
        trace = pm.sample(draws=2000,  tune=2000, step=step)


        # Set the new data
        new_GDP = GDP_POP_NCP_pred['GDP2SSP']
        new_POP = GDP_POP_NCP_pred['Value']
        new_group_idx = GDP_POP_NCP_pred['Province'].map({v:k for k,v in enumerate(UNIQUE_VALUES['Province'])})

        # Replace the observed data with new data
        pm.set_data({'GDP': new_GDP, 'POP': new_POP, 'group_idx': new_group_idx})
        # Generate posterior predictive samples
        post_pred = pm.sample_posterior_predictive(trace, var_names=['mu', 'y'])
                
            
        post_df = post_pred.posterior_predictive['y'].to_dataframe().reset_index()
        post_df = post_df.groupby(['obs_id'])[['y']].agg(['mean','std']).reset_index(drop=True)
        post_df.columns = post_df.columns.droplevel(0)
        post_df['upper'] = post_df['mean'] + 1.96 * post_df['std']
        post_df['lower'] = post_df['mean'] - 1.96 * post_df['std']
        post_df['year'] = GDP_POP_NCP_pred['year']
        post_df['Province'] = GDP_POP_NCP_pred['Province']
        post_df['SSP'] = GDP_POP_NCP_pred['SSP']


        post_df.to_csv('data/results/urban_area_pred.csv', index=False)
        post_df = pd.read_csv('data/results/urban_area_pred.csv')

        g = (plotnine.ggplot()
                + plotnine.geom_point(urban_area_hist, plotnine.aes(x='year', y='Area_cumsum_km2'), color='grey', size=0.05)
                + plotnine.geom_line(post_df, plotnine.aes(x='year', y='mean', color='SSP'), size=0.05)
                + plotnine.geom_ribbon(post_df, plotnine.aes(x='year',ymin='lower',ymax='upper', fill='SSP'), alpha=0.3)
                + plotnine.facet_wrap('~Province')
                + plotnine.theme_bw()
            )

        g = (plotnine.ggplot()
                + plotnine.geom_point(urban_area_hist, plotnine.aes(x='year', y='Area_cumsum_km2'), color='grey', size=0.05)
            )