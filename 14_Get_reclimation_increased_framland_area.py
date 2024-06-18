import math
import numpy as np
import pandas as pd
import rioxarray as rxr
import xarray as xr
import statsmodels.api as sm
import plotnine

from helper_func import read_yearbook, sample_ppf
from helper_func.calculate_GAEZ_stats import bincount_with_mask
from helper_func.parameters import BASE_YR, UNIQUE_VALUES, Monte_Carlo_num


# Read data
mask_sum = rxr.open_rasterio('data/GAEZ_v4/Province_mask.tif', chunks='auto')
mask_province = [xr.where(mask_sum == idx, 1, 0).expand_dims({'Province': [p]}) 
                      for idx,p in enumerate(UNIQUE_VALUES['Province'])]
mask_province = xr.concat(mask_province, dim='Province').astype('int8')

mask_mean = rxr.open_rasterio('data/GAEZ_v4/Province_mask_mean.tif', chunks='auto')
mask_mean_province = mask_province * mask_mean


reclimation = read_yearbook('data/Yearbook/yearbook_land_reclimation_total_area_ha.csv', 'Reclimation')     # ha
reclimation = reclimation.sort_values(['Province','year']).reset_index(drop=True)
reclimation['Cumsum Area (km2)'] = reclimation.groupby('Province')['Value'].cumsum()/1e4                    # km2


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

# Subtract values for BASE_YR to align with the beginning of the modeling period
prediction_df['mean km2'] = prediction_df.groupby('Province').transform(lambda x: x - x.iloc[0])['mean km2']
prediction_df['std km2'] = np.min(prediction_df[['std km2','mean km2']].values + 1e-5, axis=1)     # Std must > mean to avoide negative values


# Sample from mean and std of reclimation area for each province
reclimation_sample = sample_ppf(
    prediction_df['mean km2'], 
    prediction_df['std km2'], 
    n_samples=Monte_Carlo_num)

reclimation_sample = xr.DataArray(
    name='data',
    data=reclimation_sample.reshape(
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



if __name__ == '__main__':
    plotnine.options.figure_size = (6, 4)
    plotnine.options.dpi = 100

    reclimation_mean = reclimation_sample_cell.mean(dim='sample')
    reclimation_std = reclimation_sample_cell.std(dim='sample')
    reclimation_mean_stats = bincount_with_mask(mask_sum, reclimation_mean)
    reclimation_std_stats = bincount_with_mask(mask_sum, reclimation_std)
    
    reclimation_stats = reclimation_mean_stats.merge(
        reclimation_std_stats, 
        on=['bin', 'year'], 
        suffixes=('_mean', '_std'))
    
    reclimation_stats = reclimation_stats.rename(
        columns={'Value_mean':'mean', 'Value_std':'std', 'bin':'Province'})
    
    reclimation_stats['Province'] = reclimation_stats['Province'].map({
        idx:p for idx,p in enumerate(UNIQUE_VALUES['Province'])
        })
    
    reclimation_stats['lower'] = reclimation_stats['mean'] - (reclimation_stats['std'] / math.sqrt(Monte_Carlo_num) * 1.96)
    reclimation_stats['upper'] = reclimation_stats['mean'] + (reclimation_stats['std'] / math.sqrt(Monte_Carlo_num) * 1.96)
    
    g = (plotnine.ggplot(reclimation_stats) +
         plotnine.geom_line(plotnine.aes(x='year', y='mean', color='Province')) +
         plotnine.geom_ribbon(plotnine.aes(x='year', ymin='lower', ymax='upper', fill='Province'), alpha=0.5) +
         plotnine.theme_bw() +
         plotnine.labs(x='Year', y='Reclimation Area (km2)'))
    
    g.save('data/results/fig_step_14_cropland_reclimation_prediction_cumsum_km2.svg')

    