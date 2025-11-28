import xarray as xr
import rioxarray as rxr
import pandas as pd
import plotnine

from helper_func.parameters import UNIQUE_VALUES, BASE_YR


# Get masks
mask = rxr.open_rasterio('data/GAEZ_v4/GAEZ_mask.tif')
mask_mean = rxr.open_rasterio('data/GAEZ_v4/Province_mask_mean.tif')
mask_sum = rxr.open_rasterio('data/GAEZ_v4/province_mask.tif')
mask_province = [(mask_sum == idx).expand_dims({'Province': [p]}) * mask
                    for idx, p in enumerate(UNIQUE_VALUES['Province'])]
mask_province = xr.combine_by_coords(mask_province)


# Create a zero array to pad the first year of the delta data
zero_xr = xr.DataArray(0, dims=['year'], coords={'year': [BASE_YR]})

# Load the pred_to_yearbook_ratio 
pred_to_yearbook_ratio = xr.open_dataset('data/results/step_16_GAEZ_yb_base_year_production.nc', chunks='auto')['data']


# Crop yield - yearbook trend (multiplier)
yield_MC_yrbook_trend = xr.open_dataset('data/results/step_7_Yearbook_MC_ratio.nc', chunks='auto')['data']

# Crop yield - climate change (t/ha)
yield_MC_attainable_trend = xr.open_dataset('data/results/step_7_GAEZ_yield_MC.nc', chunks='auto')['data']          

# Crop area - urban encroachment (km2)
area_start = xr.open_dataset('data/results/step_15_GAEZ_area_km2_adjusted.nc', chunks='auto')['data']   # km2
area_start = area_start * mask_province
area_crop_water_ratio = area_start.sum(dim=['y', 'x', 'band']) / area_start.sum(dim=['crop', 'water_supply', 'y', 'x', 'band'])
area_crop_water_ratio = area_crop_water_ratio * mask_province       # Assign the multiplier to each pixel in the given Province
area_urban_reduce = xr.open_dataset('data/results/step_13_urban_occupy_cropland.nc', chunks='auto')['data'] # km2
area_urban_reduce = area_urban_reduce * mask_province
area_urban_reduce = area_urban_reduce * area_crop_water_ratio

# Crop area - reclamation (km2)
area_reclaim_increase = xr.open_dataset('data/results/step_14_reclimation_area_km2.nc', chunks='auto')['data'] # km2
area_reclaim_increase = area_reclaim_increase * area_crop_water_ratio




# Get the delta for each impact factor
delta_yb = yield_MC_yrbook_trend.diff('year')
delta_yb = xr.concat([zero_xr, delta_yb], dim='year')

delta_climate = yield_MC_attainable_trend.diff('year')
delta_climate = xr.concat([zero_xr, delta_climate], dim='year')

delta_urban = area_urban_reduce.diff('year')
delta_urban = xr.concat([zero_xr, delta_urban], dim='year')

delta_reclaim = area_reclaim_increase.diff('year')
delta_reclaim = xr.concat([zero_xr, delta_reclaim], dim='year')


# Calculate the relative contribution of each impact factor to the production
contr_yb = (delta_yb * yield_MC_attainable_trend) \
         * (area_start + area_reclaim_increase - area_urban_reduce) \
         * 100 / 1e6 \
         * pred_to_yearbook_ratio
    
contr_climate = delta_climate \
              * (area_start + area_reclaim_increase - area_urban_reduce) \
              * 100 / 1e6 \
              * pred_to_yearbook_ratio
              
contr_urban = - delta_urban \
            * (yield_MC_yrbook_trend * yield_MC_attainable_trend) \
            * 100 / 1e6 \
            * pred_to_yearbook_ratio
            
contr_reclaim = delta_reclaim \
                * (yield_MC_yrbook_trend * yield_MC_attainable_trend) \
                * 100 / 1e6 \
                * pred_to_yearbook_ratio
                
contr_net = contr_yb + contr_climate + contr_urban + contr_reclaim



# Get stats
contr_yb_stats = contr_yb.sum(['x', 'y']).compute().to_dataframe(name='delta').reset_index()
contr_climate_stats = contr_climate.sum(['x', 'y']).compute().to_dataframe(name='delta').reset_index()
contr_urban_stats = contr_urban.sum(['x', 'y']).compute().to_dataframe(name='delta').reset_index()
contr_reclaim_stats = contr_reclaim.sum(['x', 'y']).compute().to_dataframe(name='delta').reset_index()
contr_net_stats = contr_net.sum(['x', 'y']).compute().to_dataframe(name='delta').reset_index()


# Save the data
contr_yb_stats.to_csv('data/results/step_17_contr_yb.csv', index=False)
contr_climate_stats.to_csv('data/results/step_17_contr_climate.csv', index=False)
contr_urban_stats.to_csv('data/results/step_17_contr_urban.csv', index=False)
contr_reclaim_stats.to_csv('data/results/step_17_contr_reclaim.csv', index=False)
contr_net_stats.to_csv('data/results/step_17_contr_net.csv', index=False)


# Compute the stats for plotting
groups = ['rcp', 'SSP', 'c02_fertilization', 'crop', 'water_supply', 'Province', 'year']

def get_plot_df(in_df, groups=groups):
    in_df = in_df.copy().sort_values(groups)
    in_df = in_df.groupby(groups).describe(percentiles=[0.05, 0.5, 0.95])[['delta']].reset_index()
    in_df.columns = ['_'.join(col).strip() if col[1] != '' else col[0] for col in in_df.columns.values]
    group_obj = in_df.groupby([i for i in groups if i != 'year'])
    in_df['delta_mean_cumsum'] = group_obj[['delta_mean']].apply(lambda x: x['delta_mean'].cumsum()).values
    in_df['delta_5%_cumsum'] = group_obj[['delta_5%']].apply(lambda x: x['delta_5%'].cumsum()).values
    in_df['delta_95%_cumsum'] = group_obj[['delta_95%']].apply(lambda x: x['delta_95%'].cumsum()).values
    return in_df
    
contr_yb_stats_plot = get_plot_df(contr_yb_stats)
contr_climate_stats_plot = get_plot_df(contr_climate_stats)
contr_urban_stats_plot = get_plot_df(contr_urban_stats)
contr_reclaim_stats_plot = get_plot_df(contr_reclaim_stats,[i for i in groups if not i in ['rcp', 'SSP', 'c02_fertilization']])
contr_net_stats_plot = get_plot_df(contr_net_stats)

# Save aggregated data to csv
contr_yb_stats_plot.to_csv('data/results/step_17_contr_yb_plot.csv', index=False)
contr_climate_stats_plot.to_csv('data/results/step_17_contr_climate_plot.csv', index=False)
contr_urban_stats_plot.to_csv('data/results/step_17_contr_urban_plot.csv', index=False)
contr_reclaim_stats_plot.to_csv('data/results/step_17_contr_reclaim_plot.csv', index=False)
contr_net_stats_plot.to_csv('data/results/step_17_contr_net_plot.csv', index=False)


# Read aggregated data from csv
contr_yb_stats_plot = pd.read_csv('data/results/step_17_contr_yb_plot.csv')
contr_climate_stats_plot = pd.read_csv('data/results/step_17_contr_climate_plot.csv')
contr_urban_stats_plot = pd.read_csv('data/results/step_17_contr_urban_plot.csv')
contr_reclaim_stats_plot = pd.read_csv('data/results/step_17_contr_reclaim_plot.csv')
contr_net_stats_plot = pd.read_csv('data/results/step_17_contr_net_plot.csv')


    

if __name__ == '__main__':
    
    sel_dict = {
        'rcp': 'RCP4.5',
        # 'c02_fertilization': 'Without CO2 Fertilization',
        'SSP': 'SSP2'}
    
    sel_str = ' & '.join([f'{k}=="{v}"' for k, v in sel_dict.items()])
    group_keys = list(sel_dict.keys()) + ['c02_fertilization','year']
    
    
    # Contribution from yearbook trend
    contr_yb_total = contr_yb_stats_plot\
        .groupby(group_keys)\
        .sum(numeric_only=True)\
        .reset_index()
    contr_yb_total['type'] = 'Yearbook Trend'
    
    fig_yb = (plotnine.ggplot(contr_yb_total.query(sel_str)) +
            plotnine.geom_line(plotnine.aes(x='year', y='delta_mean_cumsum', color='c02_fertilization')) +
            plotnine.geom_ribbon(plotnine.aes(x='year', ymin='delta_5%_cumsum', ymax='delta_95%_cumsum', fill='c02_fertilization'), alpha=0.5) +
            plotnine.theme_bw() +
            plotnine.ylab('Contribution to Production (million t)') 
            )
    
    fig_yb.save('data/results/fig_step_17_contr_yb_cumsum.svg')
    
    # Contribution from climate change
    contr_climate_total = contr_climate_stats_plot\
        .groupby(group_keys)\
        .sum(numeric_only=True)\
        .reset_index()
    contr_climate_total['type'] = 'Climate Change'
        
    fig_climate = (plotnine.ggplot(contr_climate_total.query(sel_str)) +
            plotnine.geom_line(plotnine.aes(x='year', y='delta_mean_cumsum', color='c02_fertilization')) +
            plotnine.geom_ribbon(plotnine.aes(x='year', ymin='delta_5%_cumsum', ymax='delta_95%_cumsum', fill='c02_fertilization'), alpha=0.5) +
            plotnine.theme_bw() +
            plotnine.ylab('Contribution to Production (million t)')
            )
    fig_climate.save('data/results/fig_step_17_contr_climate_cumsum.svg')
    
    # Contribution from urban encroachment
    contr_urban_total = contr_urban_stats_plot\
        .groupby(group_keys)\
        .sum(numeric_only=True)\
        .reset_index()
    contr_urban_total['type'] = 'Urban Encroachment'
        
    fig_urban = (plotnine.ggplot(contr_urban_total.query(sel_str)) +
            plotnine.geom_line(plotnine.aes(x='year', y='delta_mean_cumsum', color='c02_fertilization')) +
            plotnine.geom_ribbon(plotnine.aes(x='year', ymin='delta_5%_cumsum', ymax='delta_95%_cumsum', fill='c02_fertilization'), alpha=0.5) +
            plotnine.theme_bw() +
            plotnine.ylab('Contribution to Production (million t)')
            )
    fig_urban.save('data/results/fig_step_17_contr_urban_cumsum.svg')
    
    # Contribution from reclamation
    contr_reclaim_total = contr_reclaim_stats_plot\
        .groupby( ['year'])\
        .sum(numeric_only=True)\
        .reset_index()
        
    contr_reclaim_total['type'] = 'Reclamation'
    contr_reclaim_total = contr_reclaim_total.assign(**sel_dict)
        
    fig_reclaim = (plotnine.ggplot(contr_reclaim_total) +
            plotnine.geom_line(plotnine.aes(x='year', y='delta_mean_cumsum')) +
            plotnine.geom_ribbon(plotnine.aes(x='year', ymin='delta_5%_cumsum', ymax='delta_95%_cumsum'), alpha=0.5) +
            plotnine.theme_bw() +
            plotnine.ylab('Contribution to Production (million t)')
            )
    fig_reclaim.save('data/results/fig_step_17_contr_reclaim_cumsum.svg')
    
    
    
    
    # Contribution from net
    contr_net_total = contr_net_stats_plot\
        .groupby(group_keys)\
        .sum(numeric_only=True)\
        .reset_index()
        
    fig_net = (plotnine.ggplot(contr_net_total.query(sel_str)) +
            plotnine.geom_line(plotnine.aes(x='year', y='delta_mean_cumsum', color='c02_fertilization')) +
            plotnine.geom_ribbon(plotnine.aes(x='year', ymin='delta_5%_cumsum', ymax='delta_95%_cumsum', fill='c02_fertilization'), alpha=0.5) +
            plotnine.theme_bw() +
            plotnine.ylab('Contribution to Production (million t)')
            )
    fig_net.save('data/results/fig_step_17_contr_net_cumsum.svg')
    
    
    # Contribution from all factors
    contr_total = pd.concat([contr_yb_total, contr_climate_total, contr_urban_total, contr_reclaim_total])
    fig_contr_all = (plotnine.ggplot(contr_total.query(sel_str)) +
            plotnine.geom_line(plotnine.aes(x='year', y='delta_mean_cumsum', color='c02_fertilization')) +
            plotnine.geom_ribbon(plotnine.aes(x='year', ymin='delta_5%_cumsum', ymax='delta_95%_cumsum', fill='c02_fertilization'), alpha=0.5) +
            plotnine.facet_wrap('type', nrow=2, ncol=2) +
            plotnine.theme_bw() + 
            plotnine.theme(figure_size=(14, 10)) +
            plotnine.ylab('Contribution to Production (million t)')
            )
    fig_contr_all.save('data/results/fig_step_17_contr_all_cumsum.svg')
    
    
    
    
    


