import os
import stat
import geopandas as gpd
import pandas as pd
import plotnine

from glob import glob
from tqdm.auto import tqdm
from rasterstats import zonal_stats


# Read data
region_shp = gpd.read_file('data/Vector_boundary/North_china_Plain_Province.shp')
ssp_urban_img = glob('data/Urban_1km_1km/clipped/ssp*.tiff')
urban_area_hist = pd.read_csv('data/results/area_hist_df.csv')          # unit: km2

# Loop through the images
stats = []
for img in tqdm(ssp_urban_img, total=len(ssp_urban_img)):
    base = os.path.basename(img).replace('.tiff','')
    ssp,year = base.split('_')
    statistics = zonal_stats(list(region_shp.geometry), img, stats=['sum'], nodata=-1)
    
    df = pd.DataFrame(statistics)
    df['Province'] = region_shp['EN_Name'].values
    df['year'] = int(year)
    df['ssp'] = ssp.upper()
    
    df.rename(columns={'sum': 'urban_area_km2'}, inplace=True)
    stats.append(df)
    
# Concatenate the data
stats_df = pd.concat(stats, ignore_index=True)

# Interpolate by every 5-year
stats_df['year'] = pd.to_datetime(stats_df['year'], format='%Y')
stats_df.set_index('year', inplace=True)
stats_df = stats_df.groupby(['Province', 'ssp']).resample('5YS').mean().interpolate()
stats_df.reset_index(inplace=True)
stats_df['year'] = stats_df['year'].dt.year

# Get the difference in the 2020
hist_2020 = urban_area_hist.query('year == 2019')[['Province','Area_cumsum_km2']]
ssp_2020 = stats_df.query('year == 2020')

diff = ssp_2020.merge(hist_2020, on='Province', how='left')
diff['diff'] = diff['Area_cumsum_km2'] - diff['urban_area_km2']

stats_df = stats_df.merge(diff[['Province','ssp','diff']], on=['Province','ssp'], how='left')
stats_df['urban_area_km2_adj'] = stats_df['urban_area_km2'] + stats_df['diff']



if __name__ == "__main__":
    
    
    
    g = (plotnine.ggplot() +
         plotnine.geom_point(urban_area_hist, plotnine.aes(x='year', y='Area_cumsum_km2'), color='grey', size=0.05) +
         plotnine.geom_line(stats_df.query('year >= 2020'), plotnine.aes(x='year', y='urban_area_km2_adj', color='ssp')) +
         plotnine.facet_wrap('~Province') 
         )







