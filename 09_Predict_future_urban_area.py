import pandas as pd
import pymc as pm
import arviz as az
import plotnine


# Read the hist area data
urban_area_hist = pd.read_csv('data/results/urban_area_hist.csv')
urban_area_hist = urban_area_hist.sort_values(['Province', 'year'])


# Nomalize



# Get the province codes
provinces = urban_area_hist['Province'].unique()
province_code = {v:k for k,v in enumerate(provinces)}
urban_area_hist['Province_code'] = urban_area_hist['Province'].map(province_code)


def linear_model():
    with pm.Model(coords={'Province':provinces}) as model:
        # Priors
        intercept = pm.Normal("intercept", mu=1e5-2000*500, sigma=1e11, dims="Province")
        slope = pm.Normal("slope", mu=1e3, sigma=1e3, dims="Province")
        sigma = pm.HalfNormal("sigma", 500, dims="Province")

        # Deterministic
        X = pm.Data("X", urban_area_hist['year'], dims="obs_id")
        group_idx = pm.Data("group_idx", urban_area_hist['Province_code'], dims="obs_id")
        y_obs = pm.Deterministic("y_obs", intercept[group_idx] + X * slope[group_idx], dims="obs_id")

        # Likelihood
        pm.Normal("y", mu=y_obs, sigma=sigma[group_idx], observed=urban_area_hist['Area_cumsum_km2'])

        # Sample
        idata = pm.sample()

    return model, idata



if __name__ == "__main__":

    g_hist = (plotnine.ggplot() 
              + plotnine.geom_point(urban_area_hist, plotnine.aes(x='year', y='Area_cumsum_km2', color='Province'))
    )

    ##################################
    # Linear model
    ##################################

    model, idata = linear_model()
    az.plot_trace(idata, filter_vars="regex", var_names=["~y"])


    df_posterior = idata.posterior.to_dataframe().reset_index() 

    df_posterior['year'] = urban_area_hist['year'].tolist() \
                            * df_posterior['chain'].nunique() \
                            * df_posterior['draw'].nunique()\
                            * df_posterior['Province'].nunique()


    
    df_posterior['y'] = df_posterior['intercept'] + df_posterior['year'] * df_posterior['slope']
    
    df_posterior_stats = df_posterior.groupby(['year','Province'])['y'].agg(['mean', 'std']).reset_index()
    df_posterior_stats['lower'] = df_posterior_stats['mean'] - df_posterior_stats['std'] * 1.96
    df_posterior_stats['upper'] = df_posterior_stats['mean'] + df_posterior_stats['std'] * 1.96


    g_posterior = (plotnine.ggplot(df_posterior_stats)
                     + plotnine.geom_point(plotnine.aes(x='year', y='mean', color='Province'))
                     + plotnine.geom_ribbon(plotnine.aes(x='year', ymin='lower', ymax='upper', fill='Province'), alpha=0.3)
     )