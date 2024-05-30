import ee
import re
import pandas as pd
import plotnine

from itertools import chain
from tqdm.auto import tqdm
from joblib import Parallel, delayed


# Initialize the Earth Engine
ee.Initialize()


# Import asset
SSP_Li = ee.ImageCollection("projects/sat-io/open-datasets/global-urban-extents/project_urban_scenarios")
SSP_Gao = ee.ImageCollection("projects/sat-io/open-datasets/FUTURE-URBAN-LAND/GAO_2020_2100")
NCP_shape = ee.FeatureCollection("users/wangjinzhulala/North_China_Plain_Python/Boundary_shp/North_China_Plain_province_boundry");


# Get the area
def get_area(img_id: str, shp: ee.FeatureCollection):
    img = ee.Image(img_id)
    scale_m = ee.Image(img).projection().nominalScale()
    area = img.multiply(ee.Image.pixelArea()).divide(1e6) # unit: km2
    stats = area.reduceRegions(collection=shp, reducer=ee.Reducer.sum(), scale=scale_m)
    return [i['properties'] for i in stats.getInfo()['features']]



def get_imgs(img_col: ee.ImageCollection):
    img_json = img_col.getInfo()['features']
    img_id = [i['id'] for i in img_json]
    img_name = [i['properties']['system:index'] for i in img_json]
    return img_id, img_name
    
# Get the stats
imgs, names = map(lambda x: list(chain.from_iterable(x)), zip(get_imgs(SSP_Li), get_imgs(SSP_Gao))) 
tasks = [delayed(get_area)(ee.Image(i), NCP_shape) for i in imgs]
stats = Parallel(n_jobs=-1, prefer="threads")(tqdm(tasks))

# Convert to DataFrame
stats_dfs = []
for i, s in zip(names, stats):
    df = pd.DataFrame(s)
    df['year'] = re.findall(r'\d{4}', i)[0]
    df['ssp'] = re.findall('SSP.?', i)[0]
    stats_dfs.append(df)

# Get df for each source
def interpolate(df):
    df['year'] = pd.to_datetime(df['year'], format='%Y')
    df.set_index('year', inplace=True)
    df = df.groupby(['Province', 'ssp']).resample('5YS').mean().interpolate()
    df.reset_index(inplace=True)
    df['year'] = df['year'].dt.year
    return df

stats_df = pd.concat(stats_dfs, ignore_index=True)
stats_Li = stats_df.query('SSP1.isna()').copy()
stats_Li = stats_Li[['year', 'EN_Name', 'ssp','sum']]
stats_Li = stats_Li.rename(columns={'EN_Name': 'Province', 'sum':'area_km2'})
stats_Li['ratio'] = stats_Li.groupby(['Province', 'ssp'])['area_km2'].transform(lambda x: x / x.iloc[0])
stats_Li = interpolate(stats_Li)

stats_Gao = stats_df.query('SSP1.notna()')
stats_Gao = stats_Gao[['year', 'EN_Name', 'SSP1', 'SSP2', 'SSP3', 'SSP4', 'SSP5']]
stats_Gao = stats_Gao.set_index(['year','EN_Name']).stack().reset_index()
stats_Gao = stats_Gao.rename(columns={'EN_Name': 'Province','level_2': 'ssp', 0: 'area_km2'})
stats_Gao['ratio'] = stats_Gao.groupby(['Province', 'ssp'])['area_km2'].transform(lambda x: x / x.iloc[0])
stats_Gao = interpolate(stats_Gao)


# Apply the area increase ratio to historical data
urban_area_hist = pd.read_csv('data/results/step_8_area_hist_df.csv')         
start_area = urban_area_hist.query('year == 2019')[['Province','Area_cumsum_km2']]
pred_Li_ratio = start_area.merge(stats_Li, on='Province', how='left')
pred_Li_ratio['area_km2_adj'] = pred_Li_ratio['Area_cumsum_km2'] * pred_Li_ratio['ratio']
pred_Li_ratio['Source'] = 'Li et al. (2020)'


# Apply the net increase to the historical data
Gao_2020 = stats_Gao.query('year == 2020')
diff = Gao_2020.merge(start_area, on='Province', how='left')
diff['diff'] = diff['Area_cumsum_km2'] - diff['area_km2']
pred_Gao_diff = stats_Gao.merge(diff[['Province','ssp','diff']], on=['Province','ssp'], how='left')
pred_Gao_diff['area_km2_adj'] = pred_Gao_diff['area_km2'] + pred_Gao_diff['diff']
pred_Gao_diff['Source'] = 'Gao et al. (2020)'


# Save the results
pred_Li_ratio.to_csv('data/results/step_9_Urban_SSP_pred_Li_ratio.csv', index=False)
pred_Gao_diff.to_csv('data/results/step_9_Urban_SSP_pred_Gao_diff.csv', index=False)

pred_Li_ratio = pd.read_csv('data/results/step_9_Urban_SSP_pred_Li_ratio.csv')
pred_Gao_diff = pd.read_csv('data/results/step_9_Urban_SSP_pred_Gao_diff.csv')

if __name__ == "__main__":
    
    plotnine.options.figure_size = (12, 8)
    plotnine.options.dpi = 100
    
    g = (plotnine.ggplot() +
         plotnine.geom_point(urban_area_hist, plotnine.aes(x='year', y='Area_cumsum_km2'), color='grey', size=0.05) +
         plotnine.geom_line(pred_Li_ratio, plotnine.aes(x='year', y='area_km2_adj', color='ssp', linetype='Source'), size=0.2) +
         plotnine.geom_line(pred_Gao_diff, plotnine.aes(x='year', y='area_km2_adj', color='ssp', linetype='Source'), size=0.2) +         
         plotnine.facet_wrap('~Province') +
         plotnine.theme_bw()  
         )
    

