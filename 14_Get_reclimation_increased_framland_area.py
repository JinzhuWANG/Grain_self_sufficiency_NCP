import pandas as pd
import rioxarray as rxr
import xarray as xr
import statsmodels.api as sm

from helper_func import read_yearbook, sample_ppf
from helper_func.parameters import BASE_YR, UNIQUE_VALUES, Monte_Carlo_num


# Read data
mask_mean = rxr.open_rasterio('data/GAEZ_v4/Province_mask_mean.tif', chunks='auto')
mask_mean_province = [xr.where(mask_mean == idx, 1, 0).expand_dims({'Province': [p]}) for idx,p in enumerate(UNIQUE_VALUES['Province'])]
mask_mean_province = xr.combine_by_coords(mask_mean_province).astype('float32')


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

# Make sure the order is correct
prediction_df = prediction_df.sort_values(['Province','year'])

# Sample from mean and std of reclimation area for each province
reclimation_sample = sample_ppf(
    prediction_df['mean km2'], 
    prediction_df['std km2'], 
    n_samples=Monte_Carlo_num)

reclimation_sample = xr.DataArray(
    reclimation_sample.reshape(
        Monte_Carlo_num,
        prediction_df['Province'].nunique(), 
        prediction_df['year'].nunique()),
    dims=('sample', 'Province', 'year'), 
    coords={
        'sample':range(Monte_Carlo_num), 
        'Province':prediction_df['Province'].unique(), 
        'year':prediction_df['year'].unique()})


# Distribute the reclimation area to each pixel for each province
reclimation_sample_cell = reclimation_sample * mask_mean_province
reclimation_sample_cell = reclimation_sample_cell.sum(dim='Province')


# Save to netcdf
encoding = {'data': {'zlib': True, 'complevel': 9}}
reclimation_sample_cell.name = 'data'
reclimation_sample_cell.to_netcdf('data/results/step_14_reclimation_area_km2.nc', encoding=encoding)
