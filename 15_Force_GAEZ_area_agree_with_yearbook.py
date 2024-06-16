import pandas as pd
import plotnine
import plotnine.options
import rioxarray as rxr
import xarray as xr

from helper_func.calculate_GAEZ_stats import bincount_with_mask, get_GAEZ_df, get_GAEZ_stats, get_GEAZ_layers
from helper_func.get_yearbook_records import get_yearbook_area
from helper_func.parameters import UNIQUE_VALUES



# Get the GAEZ masks
mask = rxr.open_rasterio('data/GAEZ_v4/GAEZ_mask.tif')
mask_sum = rxr.open_rasterio('data/GAEZ_v4/province_mask.tif')
mask_province = [xr.where(mask_sum == idx, 1, 0).expand_dims({'Province': [p]}) * mask
                      for idx,p in enumerate(UNIQUE_VALUES['Province'])]
mask_province = xr.concat(mask_province, dim='Province').astype('int8')



# Get data
yearbook_area = get_yearbook_area().query('year == 2020').reset_index(drop=True)
yearbook_area['area_km2'] = yearbook_area['area_yearbook_kha'] * 1000 / 100    # Convert kha to km2
yearbook_area = yearbook_area[['Province', 'crop', 'area_km2']]

GAEZ_area_stats = get_GAEZ_stats('Harvested area')\
    .sort_values(['Province', 'crop', 'water_supply' ])\
    .reset_index(drop=True)
GAEZ_area_stats['area_km2'] = GAEZ_area_stats['Harvested area'] * 1000 / 100    # Convert kha to km2



# Compare the GAEZ area with the yearbook area
GAEZ_area_sum = GAEZ_area_stats.groupby(['Province', 'crop'])[['area_km2']].sum().reset_index()
GAEZ_with_yearbook = pd.merge(GAEZ_area_sum, yearbook_area, on=['Province', 'crop'], suffixes=('_GAEZ', '_yearbook'))
GAEZ_with_yearbook['diff_ratio'] = GAEZ_with_yearbook['area_km2_yearbook'] / GAEZ_with_yearbook['area_km2_GAEZ']
GAEZ_with_yearbook_xr = xr.Dataset.from_dataframe(GAEZ_with_yearbook.set_index(['Province', 'crop'])[['diff_ratio']])
GAEZ_to_yearbook_map = GAEZ_with_yearbook_xr['diff_ratio'] * mask_province
GAEZ_to_yearbook_map = GAEZ_to_yearbook_map.sum(dim=['Province'], skipna=True).astype('float32')

# Get the GAEZ area array
GAEZ_area_df = get_GAEZ_df(var_type = 'Harvested area')
GAEZ_area_xr = get_GEAZ_layers(GAEZ_area_df)
GAEZ_area_xr = GAEZ_area_xr * 1000 / 100    # Convert kha to km2

GAEZ_area_xr_adj = GAEZ_area_xr * GAEZ_to_yearbook_map


# Save the GAEZ_area_xr_adj
GAEZ_area_xr_adj.name = 'data'
encoding = {'data': {'dtype': 'float32', 'zlib': True}}
GAEZ_area_xr_adj.to_netcdf('data/results/step_15_GAEZ_area_km2_adjusted.nc', encoding=encoding, engine='h5netcdf')




if __name__ == '__main__':
    
    plotnine.options.figure_size = (6.4, 4.8)
    plotnine.options.dpi = 100
    
    GAEZ_adj_stats = bincount_with_mask(mask_sum, GAEZ_area_xr_adj)
    GAEZ_adj_stats = GAEZ_adj_stats.rename(columns={'bin': 'Province', 'Value': 'area_km2'})
    GAEZ_adj_stats['Province'] = GAEZ_adj_stats['Province'].map({
        idx:p for idx,p in enumerate(UNIQUE_VALUES['Province'])
        })
    
    GAEZ_adj_stats_sum = GAEZ_adj_stats.groupby(['Province','crop'])['area_km2'].sum().reset_index()
    GAEZ_adj_with_yearbook = pd.merge(
        GAEZ_adj_stats_sum, 
        yearbook_area, 
        on=['Province','crop'], 
        suffixes=('_GAEZ', '_yearbook')
    )
    
    g = (plotnine.ggplot(GAEZ_adj_with_yearbook) +
            plotnine.geom_point(plotnine.aes(x = 'area_km2_GAEZ', y = 'area_km2_yearbook', color = 'crop')) +
            plotnine.geom_abline(intercept = 0, slope = 1, linetype='dashed', size=0.2) +
            plotnine.facet_wrap('~Province') +
            plotnine.theme_bw()
            )
    
    






