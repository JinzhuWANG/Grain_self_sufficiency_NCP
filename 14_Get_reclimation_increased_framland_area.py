import pandas as pd
import xarray as xr
import statsmodels.api as sm

from helper_func import read_yearbook
from helper_func.parameters import BASE_YR


# Read data
reclimation = read_yearbook('data/Yearbook/yearbook_land_reclimation_total_area_ha.csv', 'Reclimation') # ha
reclimation = reclimation.sort_values(['Province','year']).reset_index(drop=True)
reclimation['Cumsum Area (km2)'] = reclimation.groupby('Province')['Value'].cumsum()/1e4                # km2


# Build model for each province
prediction_df = pd.DataFrame()
for idx, df in reclimation.groupby('Province'):

    # Fit model
    X = sm.add_constant(df['year'])
    y = df['Cumsum Area (km2)']
    model = sm.OLS(y, X).fit()
    
    # Predict
    pred_df = pd.DataFrame({'year':range(BASE_YR,2101)})
    pred_df = sm.add_constant(pred_df)

    inf_df = model.get_prediction(pred_df).summary_frame(alpha=0.32) # 0.68 CI indicates the mean+/-std

    pred_df['Province'] = idx
    pred_df['mean km2'] = inf_df['mean']
    pred_df['std km2'] = (inf_df['obs_ci_upper'] - inf_df['mean'])
    pred_df = pred_df.drop(['const'], axis=1)

    prediction_df = pd.concat([prediction_df, pred_df])




