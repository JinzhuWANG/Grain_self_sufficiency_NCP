from itertools import product
import pandas as pd
import plotnine
from helper_func.parameters import Province_names_en

# read GYGA (global yield gap atlas) data
GYGA = pd.read_csv('data/GYGA/GygaChina_Crop_potential_yield.csv')



#############################################
#               Data cleaning               #
#############################################

# split station_name into city_name and province_name
GYGA['City'] = GYGA['STATIONNAME'].apply(lambda x:x.split('(')[0].strip())
GYGA['Province'] = GYGA['STATIONNAME'].apply(lambda x:x.split('(')[1][:-1].strip())
# split CROP into CROP and Irrigation
GYGA['crop'] = GYGA['CROP'].apply(lambda x:x.split(' ')[1].strip())
GYGA['water_supply'] = GYGA['CROP'].apply(lambda x:x.split(' ')[0].strip())


# filter the rows that area in the research provinces
GYGA = GYGA[GYGA['Province'].isin(Province_names_en)].reset_index(drop=True)
# remove unnucessay cols
GYGA = GYGA[['Province','crop','water_supply','YW','YP']]


# rename crop to match GAEZ records
crop_rename_dict = {'rice':'Wetland rice','wheat':'Wheat','maize':'Maize'}
GYGA = GYGA.replace(crop_rename_dict)


# divide GYGA into rainfed/irrigated
GYGA_rainfed = GYGA[GYGA['water_supply']=='Rainfed'].rename(columns={'YW':'yield_potential'}).drop(columns='YP')
GYGA_irrigated = GYGA[GYGA['water_supply']=='Irrigated'].rename(columns={'YP':'yield_potential'}).drop(columns='YW')

# concat GYGA data
GYGA_YP = pd.concat([GYGA_rainfed,GYGA_irrigated]).reset_index(drop=True)


#############################################
#          Fill the missing values          #
#############################################

# Get all unique values of 'Province', 'crop', and 'water_supply'
provinces = GYGA_YP['Province'].unique()
crops = GYGA_YP['crop'].unique()
water_supplies = GYGA_YP['water_supply'].unique()

# Create a DataFrame of all possible combinations
all_combinations = pd.DataFrame(list(product(provinces, crops, water_supplies)), columns=['Province', 'crop', 'water_supply'])

# Merge with the original DataFrame
merged = pd.merge(all_combinations, GYGA_YP, how='left', on=['Province', 'crop', 'water_supply'])

# Fill missing values with the mean of the group defined by 'crop' and 'water_supply'
merged['yield_potential'] = merged.groupby(['crop', 'water_supply'])['yield_potential'].transform(lambda x: x.fillna(x.mean()))

# If there are still NaN values (i.e., some combinations of 'crop' and 'water_supply' were not present in the original DataFrame), 
# then fill it with 0s.
merged['yield_potential'] = merged['yield_potential'].fillna(0)

# Save the filled data
merged.to_csv('data/GYGA/GYGA_attainable_filled.csv',index=False)


# plot the filled GYGA data
plotnine.options.figure_size = (10, 5)
plotnine.options.dpi = 100

g = (plotnine.ggplot(merged,plotnine.aes(x='Province',y='yield_potential')) +
 plotnine.geom_boxplot() +
 plotnine.facet_grid('water_supply~crop') +
 plotnine.theme_bw(base_size=11) +
 plotnine.theme(axis_text_x=plotnine.element_text(rotation=60))
)
