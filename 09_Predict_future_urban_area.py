import numpy as np
import pandas as pd
import pymc as pm
import arviz as az


# Read the hist area data
urban_area_hist = pd.read_csv('data/results/urban_area_hist.csv')


# Convert categorical variable into integer
urban_area_hist['Province_code'] = urban_area_hist['Province'].astype('category').cat.codes

# Compute the mean and std for each province
means = urban_area_hist.groupby('Province_code')['Area_cumsum_km2'].mean().to_dict()
stds = urban_area_hist.groupby('Province_code')['Area_cumsum_km2'].std().to_dict()


# Normalize the data by province
urban_area_hist['Area_cumsum_normalized'] = urban_area_hist.apply(lambda row: (row['Area_cumsum_km2'] - means[row['Province_code']]) / stds[row['Province_code']], axis=1)



# Define the model
coords = {"group": urban_area_hist['Province'].unique()}


# Define the model
def create_bayesian_model():
    # Hyperpriors
    intercept_mu = pm.Normal("intercept_mu", 0, sigma=5)
    intercept_sigma = pm.HalfNormal("intercept_sigma", sigma=5)
    slope_mu = pm.Normal("slope_mu", 0, sigma=5)
    slope_sigma = pm.HalfNormal("slope_sigma", sigma=5)
    sigma_hyperprior = pm.HalfNormal("sigma_hyperprior", sigma=5)

    # Define priors
    sigma = pm.HalfNormal("sigma", sigma=sigma_hyperprior, dims="group")

    β0_offset = pm.Normal("β0_offset", 0, sigma=5, dims="group")
    β0 = pm.Deterministic("β0", intercept_mu + β0_offset * intercept_sigma, dims="group")
    β1_offset = pm.Normal("β1_offset", 0, sigma=5, dims="group")
    β1 = pm.Deterministic("β1", slope_mu + β1_offset * slope_sigma, dims="group")

    # Data
    x = pm.Data("x", urban_area_hist['year'], dims="obs_id")
    g = pm.Data("g", urban_area_hist['Province_code'], dims="obs_id")
    # Linear model
    μ = pm.Deterministic("μ", β0[g] + β1[g] * x, dims="obs_id")
    # Define likelihood
    pm.Normal("y", mu=μ, sigma=sigma[g], observed=urban_area_hist['Area_cumsum_normalized'], dims="obs_id")



def main():
    with pm.Model(coords=coords) as hierarchical:
        create_bayesian_model()
        return pm.sample(return_inferencedata=True, progressbar=True)
        
    


if __name__ == '__main__':
    idata = main()
    az.plot_trace(idata, filter_vars="regex", var_names=["~μ"])



