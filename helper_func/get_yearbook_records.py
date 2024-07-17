import re
import pandas as pd
from helper_func import read_yearbook


def get_yearbook_yield():
    """
    Retrieves the historical yield data from the yearbook for wheat, wetland rice, and maize.

    Returns:
        pandas.DataFrame: A DataFrame containing the historical yield data.
        The DataFrame has the following columns:
          - 'Year': The year of the yield data.
          - 'Province': The province where the yield data is recorded.
          - 'Crop': The crop for which the yield data is recorded.
          - 'Value': The yield value in kilograms.
          - 'Yield (tonnes)': The yield value converted to tonnes.
    """
    # Read the yearbook data for wheat, wetland rice, and maize
    wheat_yield_history = read_yearbook('data/Yearbook/Provincial_wheat_yield.csv','Wheat')
    rice_yield_history = read_yearbook('data/Yearbook/Provincial_rice_yield.csv','Wetland rice')
    maize_yield_history = read_yearbook('data/Yearbook/Provincial_maize_yield.csv','Maize')

    # Concatenate the data, and convert kg to tonnes
    yearbook_yield = pd.concat([wheat_yield_history, rice_yield_history, maize_yield_history], axis=0)
    yearbook_yield['Yield (tonnes)'] = yearbook_yield['Value'] / 1000
    return yearbook_yield



def get_yearbook_area():
    # Read the yearbook data
    wheat_area_history = read_yearbook('data/Yearbook/Area_wheat.csv','Wheat')
    rice_area_history = read_yearbook('data/Yearbook/Area_rice.csv','Wetland rice')
    maize_area_history = read_yearbook('data/Yearbook/Area_maize.csv','Maize')

    # Concatenate the data, and convert ha to kha
    yearbook_area = pd.concat([wheat_area_history, rice_area_history, maize_area_history], axis=0)
    yearbook_area = yearbook_area.rename(columns={'Value':'area_yearbook_kha'})
    
    # Calculate the ratio of the area to the total area
    yearbook_area = yearbook_area.sort_values(['Province','year', 'crop'])
    yearbook_area['area_ratio'] = yearbook_area\
        .groupby(['Province','year'])['area_yearbook_kha']\
        .transform(lambda x: x / x.sum())

    return yearbook_area


def get_yearbook_production():
    # Read the yearbook data
    wheat_production = read_yearbook('data/Yearbook/production_wheat.csv','Wheat')
    maize_productoin = read_yearbook('data/Yearbook/production_maize.csv','Maize')
    rice_production = read_yearbook('data/Yearbook/production_rice.csv','Wetland rice') 
    
    # Concatenate the data, and convert tonnes to million tonnes
    yearbook_production = pd.concat([wheat_production, maize_productoin, rice_production], axis=0)
    yearbook_production['Production (Mt)'] = yearbook_production['Value'] / 100
    
    return yearbook_production


def get_China_GDP():
    GDP_China = pd.read_csv('data/Yearbook/yearbook_GDP_China.csv').T.drop('地区')                                      # unit: 10k CNY
    GDP_China = GDP_China.sum(axis=1).reset_index().rename(columns={'index':'year', 0:'GDP'})
    GDP_China['year'] = GDP_China['year'].astype('int16')
    GDP_China['GDP'] = GDP_China['GDP'].astype('float64')   
    return GDP_China

def get_China_population():
    Population_China = pd.read_excel('data/Yearbook/China_population_1980_2022.xlsx', sheet_name='统计数据')             # unit: 10k person
    Population_China = Population_China[Population_China["地区名称"] != '中国']
    Population_China = Population_China.groupby(['统计年度'])[['总人口数/万人']].sum().reset_index()
    Population_China.columns = ['year','Population']
    Population_China['Population'] = Population_China['Population'].astype('float64')
    Population_China['Population'] = Population_China['Population']/1e2                                                 # unit: million person
    return Population_China

def get_NCP_urban_population_ratio():
    urban_pop_ratio = read_yearbook('data/Yearbook/Urban_population_rate.csv')  # unit: %
    return urban_pop_ratio[['Province', 'year', 'Value']].rename(columns={'Value':'urban_pop_ratio'})