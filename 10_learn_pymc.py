from cgi import test
import arviz as az
from matplotlib.pylab import beta
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import xarray as xr
import plotnine


rng = np.random.default_rng(1234)


group_list = ["one", "two", "three", "four", "five"]
trials_per_group = 20

group_intercepts = rng.normal(0, 1, len(group_list))
group_slopes = np.ones(len(group_list)) * -0.5
group_mx = group_intercepts * 2

group = np.repeat(group_list, trials_per_group)
subject = np.repeat(np.arange(len(group_list)), trials_per_group)
intercept = np.repeat(group_intercepts, trials_per_group)
slope = np.repeat(group_slopes, trials_per_group)
mx = np.repeat(group_mx, trials_per_group)

x = rng.normal(mx, 1)
y = rng.normal(intercept + (x - mx) * slope, 1)

data = pd.DataFrame({"group": group, "group_idx": subject, "x": x, "y": y})


def get_linear():
    with pm.Model() as model:
        # Priors
        intercepts = pm.Normal("intercepts", 0, 5)
        slopes = pm.Normal("slopes", 0, 5)
        sigma = pm.HalfNormal("sigma", 5)

        # Deterministic
        X = pm.MutableData("X", data["x"], dims="obs_id")
        pred = pm.Deterministic("pred", intercepts + X * slopes)

        # Likelihood
        pm.Normal("y", mu=pred, sigma=sigma, observed=data["y"], dims="obs_id")

        # Sample
        idata = pm.sample()

    return model, idata



def get_linear_group():
    coords = {"group": group_list}

    with pm.Model(coords=coords) as model:
        # Priors
        intercepts = pm.Normal("intercepts", 0, 5, dims="group")
        slopes = pm.Normal("slopes", 0, 5, dims="group")
        sigma = pm.HalfNormal("sigma", 5, dims="group")
        # sigma = pm.HalfCauchy("sigma", beta=2, dims="group")
               
        X = pm.MutableData("X", data["x"], dims="obs_id")
        group_idx = pm.MutableData("group_idx", data["group_idx"], dims="obs_id")
        
        # Deterministic
        pred = pm.Deterministic("pred", intercepts[group_idx] + X * slopes[group_idx])

        # Likelihood
        pm.Normal("y", mu=pred, sigma=sigma[group_idx], observed=data["y"], dims="obs_id")

        # Sample
        idata = pm.sample()

    return model, idata



def create_test_model():
    
    # True parameters
    true_slope = 100
    true_intercept = 50

    # Generate some data
    np.random.seed(42)
    x = 10 * np.random.rand(100)
    y = true_slope * x + true_intercept + np.random.normal(0, 20, size=x.shape)
    
    # Create the model
    with pm.Model() as model:
        # Priors for unknown model parameters
        slope = pm.Normal("slope", mu=1, sigma=10)
        intercept = pm.Normal("intercept", mu=1, sigma=10)
        
        # Likelihood (sampling distribution) of observations
        sigma = pm.HalfCauchy("sigma", beta=5)
        y_obs = pm.Normal("y_obs", mu=slope * x + intercept, sigma=sigma, observed=y)
        
        # Posterior distribution
        trace = pm.sample(1000)
        
    return model, trace






if __name__ == '__main__':

    model_linear, idata = get_linear()
    model_liearn_group, idata_group = get_linear_group()

    #################################################
    # Linear model
    #################################################

    # Posterior predictive
    with model_linear:
        x = np.linspace(data["x"].min(), data["x"].max(), 100)
        pm.set_data({"X": x})
        idata.extend(pm.sample_posterior_predictive(idata, var_names=["pred",'y']))

    # Get conditional mean data
    idata_mean_summary = az.summary(idata.posterior_predictive['pred'], hdi_prob=0.95).reset_index()
    idata_mean_summary['x'] = x

    idata_hdi_summary = az.summary(idata.posterior_predictive['y'], hdi_prob=0.95).reset_index()
    idata_hdi_summary['x'] = x

    g_mean = (plotnine.ggplot() +
            plotnine.geom_point(data, plotnine.aes(x="x", y="y", color="group")) +
            plotnine.geom_ribbon(idata_mean_summary, plotnine.aes(x="x", ymin="hdi_2.5%", ymax='hdi_97.5%'),alpha=0.3) 
            )
        
    g_hdi = (plotnine.ggplot() +
            plotnine.geom_point(data, plotnine.aes(x="x", y="y", color="group")) +
            plotnine.geom_ribbon(idata_hdi_summary, plotnine.aes(x="x", ymin="hdi_2.5%", ymax='hdi_97.5%'),alpha=0.3) 
            )

    #################################################
    # Linear model with group
    #################################################

    # Posterior predictive
    with model_liearn_group:
        x = [
            np.linspace(data.query(f"group_idx=={i}").x.min(), data.query(f"group_idx=={i}").x.max(), 20)
            for i, _ in enumerate(group_list)
        ]
        g = [np.ones(20) * i for i, _ in enumerate(group_list)]

        x = np.concatenate(x)
        g = np.concatenate(g)

        pm.set_data({"X": x, "group_idx": g.astype(int)})
        idata_group.extend(pm.sample_posterior_predictive(idata_group, var_names=["pred",'y']))

    # Get conditional mean data
    idata_mean_summary = az.summary(idata_group.posterior_predictive['pred'], hdi_prob=0.95).reset_index()
    idata_mean_summary['x'] = x
    idata_mean_summary['group'] = data['group']

    idata_hdi_summary = az.summary(idata_group.posterior_predictive['y'], hdi_prob=0.95).reset_index()
    idata_hdi_summary['x'] = x
    idata_hdi_summary['group'] = data["group"]

    g_mean_group = (plotnine.ggplot() +
            plotnine.geom_point(data, plotnine.aes(x="x", y="y", color="group")) +
            plotnine.geom_ribbon(idata_mean_summary, plotnine.aes(x="x", ymin="hdi_2.5%", ymax='hdi_97.5%', fill='group'),alpha=0.3) 
            )
    
    g_hdi_group = (plotnine.ggplot() +
            plotnine.geom_point(data, plotnine.aes(x="x", y="y", color="group")) +
            plotnine.geom_ribbon(idata_hdi_summary, plotnine.aes(x="x", ymin="hdi_2.5%", ymax='hdi_97.5%', fill='group'),alpha=0.3) 
            )


    
    #################################################
    # Test model
    #################################################
    
    # Plotting the results
    test_model, test_trace = create_test_model()
    az.plot_trace(test_trace, filter_vars="regex", var_names=["~y_obs"])




