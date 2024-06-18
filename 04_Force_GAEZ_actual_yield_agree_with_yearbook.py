import pandas as pd
import plotnine
import rioxarray as rxr
import xarray as xr

from helper_func.calculate_GAEZ_stats import bincount_with_mask, get_GAEZ_df, get_GAEZ_stats
from helper_func.get_yearbook_records import get_yearbook_yield
from helper_func.parameters import UNIQUE_VALUES


# Get data
yearbook_yield = get_yearbook_yield().query('year == 2020').reset_index(drop=True)
GAEZ_area_stats = get_GAEZ_stats('Harvested area').sort_values(['Province', 'crop', 'water_supply' ]).reset_index(drop=True)
GAEZ_area_stats['area_ratio'] = GAEZ_area_stats.groupby(['Province', 'crop'])[['Harvested area']].transform(lambda x: x / x.sum())

# Get the GAEZ_yield stats
mask = rxr.open_rasterio('data/GAEZ_v4/GAEZ_mask.tif') 
mask_mean = rxr.open_rasterio('data/GAEZ_v4/province_mask_mean.tif')
mask_sum = rxr.open_rasterio('data/GAEZ_v4/province_mask.tif')

# Compute the GAEZ_yield stats
GAEZ_yield_df = get_GAEZ_df(var_type = 'Yield')

xr_arrs = []
for _, row in GAEZ_yield_df.iterrows():
    path = row['fpath']
    crop = row['crop']
    water_supply = row['water_supply']

    xr_arr = rxr.open_rasterio(path).squeeze()
    xr_arr = xr_arr.expand_dims({'crop': [crop], 'water_supply': [water_supply]})
    xr_arrs.append(xr_arr)

GAEZ_yield_xr = xr.combine_by_coords(xr_arrs)

GAEZ_yield_stats = bincount_with_mask(mask_sum, GAEZ_yield_xr * mask_mean)
GAEZ_yield_stats = GAEZ_yield_stats.rename(columns = {'bin': 'Province', 'Value': 'Yield (t/ha)'})
GAEZ_yield_stats['Province'] = GAEZ_yield_stats['Province'].map(dict(enumerate(UNIQUE_VALUES['Province'])))


# Merge the GAEZ_yield_stats with the GAEZ_area_stats
GAEZ_stats = pd.merge(GAEZ_yield_stats, GAEZ_area_stats, on = ['Province', 'crop', 'water_supply'])
GAEZ_stats['Yield_weighted'] = GAEZ_stats['Yield (t/ha)'] * GAEZ_stats['area_ratio']
GAEZ_stats = GAEZ_stats.groupby(['crop','Province'])[['Yield_weighted']].sum().reset_index()


# Compute the ratio of GAEZ to yearbook
GAEZ_yb = pd.merge(GAEZ_stats, yearbook_yield, on = ['crop', 'Province'], suffixes = ('_GAEZ', '_yearbook'))
GAEZ_yb['GAEZ_to_yb_multiplier'] = GAEZ_yb['Yield (tonnes)'] / GAEZ_yb['Yield_weighted']
GAEZ_yb = GAEZ_yb.set_index(['crop', 'Province'])[['GAEZ_to_yb_multiplier']]
GAEZ_yb_xr = xr.Dataset.from_dataframe(GAEZ_yb)['GAEZ_to_yb_multiplier']

# Distribute the multiplier to each province
mask_province = [(mask_sum == idx).expand_dims({'Province': [p]}) 
                 for idx,p in enumerate(UNIQUE_VALUES['Province'])]
mask_province = xr.combine_by_coords(mask_province)
mask_province = mask_province * GAEZ_yb_xr            # ratio/pixel


# Multiply the GAEZ_yield_xr by the multiplier
GAEZ_yield_xr_adj = GAEZ_yield_xr * mask_province
GAEZ_yield_xr_adj = GAEZ_yield_xr_adj.sum(dim=['Province'])

GAEZ_yield_xr_adj.name = 'data'
encoding = {'data': {'dtype': 'float32', 'zlib': True}}
GAEZ_yield_xr_adj.to_netcdf('data/results/step_4_GAEZ_actual_yield_adj.nc', encoding=encoding, engine='h5netcdf')


# Sanity check
if __name__ == '__main__':

    # Get the stats
    GAEZ_yield_adj_stats = bincount_with_mask(mask_sum, GAEZ_yield_xr_adj * mask_mean)           
    GAEZ_yield_adj_stats = GAEZ_yield_adj_stats.rename(columns = {'bin': 'Province', 'Value': 'Yield (t/ha)'})
    GAEZ_yield_adj_stats['Province'] = GAEZ_yield_adj_stats['Province'].map(dict(enumerate(UNIQUE_VALUES['Province'])))

    # Merge the GAEZ_yield_adj_stats with the GAEZ_area_stats
    GAEZ_adj_stats = pd.merge(GAEZ_yield_adj_stats, GAEZ_area_stats)
    GAEZ_adj_stats['Yield_weighted'] = GAEZ_adj_stats['Yield (t/ha)'] * GAEZ_adj_stats['area_ratio']
    GAEZ_adj_stats = GAEZ_adj_stats.groupby(['crop','Province'])[['Yield_weighted']].sum().reset_index()
    
    GAEZ_adj_yb_stats = pd.merge(GAEZ_adj_stats, yearbook_yield, on = ['crop', 'Province'], suffixes = ('_GAEZ', '_yearbook'))

    plotnine.options.figure_size = (10, 6)
    plotnine.options.dpi = 100
    g = (plotnine.ggplot(GAEZ_adj_yb_stats) +
            plotnine.geom_point(plotnine.aes(x = 'Yield (tonnes)', y = 'Yield_weighted', color = 'crop')) +
            plotnine.geom_abline(intercept = 0, slope = 1, linetype='dashed', size=0.2) +
            plotnine.facet_wrap('~Province') +
            plotnine.theme_bw()
            )
    
    g.save('data/results/fig_step_4_GAEZ_actual_yield_agree_with_yearbook_kg_ha.svg')



