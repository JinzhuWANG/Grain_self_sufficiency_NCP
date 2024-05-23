import pandas as pd
from helper_func import read_yearbook


def get_yearbook_yield():
    # Read the yearbook dataWetland rice
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
    return yearbook_area