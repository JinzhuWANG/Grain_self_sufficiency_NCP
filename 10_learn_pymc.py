import arviz as az
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




if __name__ == '__main__':

    with pm.Model() as linear:
        # Priors
        intercepts = pm.Normal("intercepts", 0, 5)
        slopes = pm.Normal("slopes", 0, 5)
        sigma = pm.HalfNormal("sigma", 5)

        # Deterministic
        X = pm.MutableData("X", data["x"], dims="obs_id")
        
        # Linear model
        pred = pm.Deterministic("pred", intercepts + X * slopes)

        # Likelihood
        pm.Normal("y", mu=pred, sigma=sigma, observed=data["y"], dims="obs_id")


    with linear:
        idata = pm.sample()


    with linear:
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





########################################
#       Model with group effects       #
########################################

if __name__ == '__main__':

    coords = {"group": group_list}

    with pm.Model(coords=coords) as linear_group:
        # Priors
        intercepts = pm.Normal("intercepts", 0, 5, dims="group")
        slopes = pm.Normal("slopes", 0, 5, dims="group")
        sigma = pm.HalfNormal("sigma", 5, dims="group")
        
        
        # Deterministic
        X = pm.MutableData("X", data["x"], dims="obs_id")
        group_idx = pm.MutableData("group_idx", data["group_idx"], dims="obs_id")

        # Linear model
        pred = pm.Deterministic("pred", intercepts[group_idx] + X * slopes[group_idx])

        # Likelihood
        pm.Normal("y", mu=pred, sigma=sigma[group_idx], observed=data["y"], dims="obs_id")







