import pandas as pd
import numpy as np

import pymc as pm

from helper_func.parameters import UNIQUE_VALUES


# Read data
pop_total_and_ratio = pd.read_csv('data/results/step_9_2_total_pop_and_urban_ratio.csv')
urban_hist_area_km2 = pd.read_csv('data/results/step_9_1_urban_area_ext.csv')
urban_hist_area_km2['area_mornal'] = urban_hist_area_km2\
    .groupby('Province')['Area_cumsum_km2']\
    .transform(lambda x: (x - x.mean()) / x.std())


# Combine the data
data_hist_normal = pd.merge(
    pop_total_and_ratio,
    urban_hist_area_km2,
    on=['Province', 'year']
)[['Province', 'year', 'pop_mornal', 'ratio_mornal', 'area_mornal']]

data_hist_normal['Province_idx'] = data_hist_normal['Province']\
    .map(lambda x: {v:k for k,v in dict(enumerate(UNIQUE_VALUES['Province'])).items()}[x])


# Build mc model
if __name__ == '__main__':
    
    with pm.Model(coords={'Province_idx':UNIQUE_VALUES['Province']}) as ind_slope_intercept:
        
        # Priors
        sigma = pm.HalfCauchy('sigma', beta=2, dims='Province_idx')
        intercepts = pm.Normal('intercepts', mu=0, sigma=1, dims='Province_idx')
        slopes_pop = pm.Normal('slopes_pop', mu=0, sigma=1, dims='Province_idx')
        slopes_ratio = pm.Normal('slopes_ratio', mu=0, sigma=1, dims='Province_idx')
        
        
        # Data
        pop_mornal = pm.Data('pop_mornal', data_hist_normal['pop_mornal'], dims='obs_id')
        ratio_mornal = pm.Data('ratio_mornal', data_hist_normal['ratio_mornal'], dims='obs_id')
        province = pm.Data('Province_idx', data_hist_normal['Province_idx'], dims='obs_id')
        
        # Linear model
        mu = pm.Deterministic(
            'mu', 
            intercepts[province] + (slopes_pop[province] * pop_mornal) + (slopes_ratio[province] * ratio_mornal),
            dims='obs_id')
        
        # Likelihood
        pm.Normal('y', mu=mu, sigma=sigma[province], observed=data_hist_normal['area_mornal'], dims='obs_id')
        
        # Sample
        trace = pm.sample(2000, cores=4)