import numpy as np
import itertools
import pandas as pd
import pymc as pm
import arviz as az
import plotnine


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
    x = pm.MutableData("x", urban_area_hist['year'], dims="obs_id")
    g = pm.MutableData("g", urban_area_hist['Province_code'], dims="obs_id")
    # Linear model
    μ = pm.Deterministic("μ", β0[g] + β1[g] * x, dims="obs_id")
    # Define likelihood
    pm.Normal("y", mu=μ, sigma=sigma[g], observed=urban_area_hist['Area_cumsum_normalized'], dims="obs_id")



    

if __name__ == '__main__':

    coords = {"group": urban_area_hist['Province'].unique()}

    # Define the model
    with pm.Model(coords=coords) as hierarchical:
        create_bayesian_model()
        step = pm.NUTS(target_accept=0.95, max_treedepth=15)
        idata =  pm.sample(step=step, 
                         return_inferencedata=True, 
                         tune=2000, 
                         draws=2000)
        

    az.plot_trace(idata, filter_vars="regex", var_names=["~μ"])




# posterior prediction for these x values
xi = urban_area_hist['year'].values
gi = urban_area_hist['Province_code'].values

# do posterior predictive inference
with hierarchical:
    pm.set_data({"x": xi, "g": gi})
    idata.extend(pm.sample_posterior_predictive(idata, var_names=["y", "μ"]))



posterior_df = pd.DataFrame({'value': idata.posterior_predictive.y.values.flatten()},
                            index=pd.MultiIndex.from_product(
                                [idata.posterior_predictive.chain.values.flatten(), 
                                 idata.posterior_predictive.draw.values.flatten(),
                                 urban_area_hist['year'].unique(),
                                 urban_area_hist['Province'].unique()],
                                names=['chain', 'draw','year', 'Province']
                                )   
                            ).reset_index()

posterior_stats = posterior_df.groupby(['year', 'Province'])['value'].agg(['mean', 'std']).reset_index()                             
posterior_stats['lower'] = posterior_stats['mean'] - 2 * posterior_stats['std']
posterior_stats['upper'] = posterior_stats['mean'] + 2 * posterior_stats['std']

# Plot the posterior predictive
g = (plotnine.ggplot() +
        plotnine.geom_line(urban_area_hist,plotnine.aes(x='year', y='Area_cumsum_normalized', color='Province')) +
        # plotnine.geom_ribbon(posterior_stats, plotnine.aes(x='year',ymin='lower',ymax='upper', fill='Province'), alpha=0.5) +
        plotnine.geom_point(posterior_stats, plotnine.aes(x='year', y='mean', color='Province'), size=0.2, alpha=0.3) +
        plotnine.ggtitle('Posterior predictive') +
        plotnine.ylab('Normalized area') +
        plotnine.xlab('Year')
    )



