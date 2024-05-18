import re
from helper_func.calculate_GAEZ_stats import get_GAEZ_stats
from helper_func.get_yearbook_records import get_yearbook_area, get_yearbook_yield
from helper_func.parameters import BASE_YR


# Get the GAEZ stats
GAEZ_area_stats = get_GAEZ_stats('Harvested area')\
    .sort_values(['Province','crop','water_supply'])                # kha

GAEZ_area_stats['area_ratio'] = GAEZ_area_stats\
    .groupby(['Province', 'crop'])['Harvested area']\
    .transform(lambda x: x / x.sum())

GAEZ_yield_stats = get_GAEZ_stats('Yield')                          # t/ha

# Merge the Area and Yield
GAEZ_area_yield = GAEZ_area_stats.merge(GAEZ_yield_stats)
GAEZ_area_yield['Yield_weighted'] = GAEZ_area_yield['Yield'] * GAEZ_area_yield['area_ratio']

# Get the Yearbook records
yearbook_yield = get_yearbook_yield().query(f'year == {BASE_YR}')   # t/ha
yearbook_area = get_yearbook_area().query(f'year == {BASE_YR}')     # kha



def get_GAEZ_yield_multiplier():
    GAEZ_yield_province = GAEZ_area_yield.groupby(['Province', 'crop'])['Yield_weighted'].sum().reset_index()
    # Get the ratio of Yearbook_yield / GAEZ_yield in the base year
    GAEZ2YB_yield_multiplier = yearbook_yield.merge(GAEZ_yield_province, on=['Province', 'crop'])
    GAEZ2YB_yield_multiplier['ratio'] = GAEZ2YB_yield_multiplier['Yield (tonnes)'] / GAEZ2YB_yield_multiplier['Yield_weighted']
    return GAEZ2YB_yield_multiplier[['Province', 'crop', 'ratio']]


def get_GAEZ_area_multiplier():
    GAEZ_area_province = GAEZ_area_yield.groupby(['Province', 'crop'])['Harvested area'].sum().reset_index()
    # Get the ratio of Yearbook_yield / GAEZ_area in the base year
    GAEZ2YB_area_multiplier = yearbook_area.merge(GAEZ_area_province, on=['Province', 'crop'])
    GAEZ2YB_area_multiplier['ratio'] = GAEZ2YB_area_multiplier['area_yearbook_kha'] / GAEZ2YB_area_multiplier['Harvested area']
    return GAEZ2YB_area_multiplier[['Province', 'crop', 'ratio']]
