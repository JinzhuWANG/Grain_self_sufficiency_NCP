# Define the GAEZ input data
GAEZ_input = ['GAEZ_4',
              'GAEZ_5']

# The base year 
BASE_YR = 2020
TARGET_YR = 2100

# Define the  crops, water_supply, and c02_fertilization used in the analysis
UNIQUE_VALUES = { 
            'crop':['Maize', 'Wetland rice', 'Wheat'],
            'water_supply':['Dryland', 'Irrigated'],
            'c02_fertilization':['With CO2 Fertilization', 'Without CO2 Fertilization'],
            'rcp':['RCP2.6', 'RCP4.5', 'RCP6.0', 'RCP8.5'],
            'Province':['Anhui', 'Beijing', 'Hebei', 'Henan', 'Jiangsu', 'Shandong', 'Tianjin'],
            'attainable_year':list(range(2010, 2101, 5)),
            'simulation_year':list(range(BASE_YR, 2101, 5))
}

DIM_ABBRIVATION = {'c':'crop', 
                   's':'water_supply', 
                   'r':'rcp', 
                   'o':'c02_fertilization', 
                   'p':'Province', 
                   'h':'height', 
                   'w':'width',
                   'y':'simulation_year'}


# The province in lexciographical order
Province_names_cn_en = dict(zip(
    ['北京市','天津市', '河北省','江苏省', '安徽省', '河南省', '山东省'],
    ['Beijing','Tianjin', 'Hebei','Jiangsu', 'Anhui', 'Henan', 'Shandong']
))


# Define the number of parallel threads to use
PARALLEY_THREADS = 32


# Define the full names of the GAEZ categories
GAEZ_full_names = {
    'GAEZ_1':'Land and Water Resources',
    'GAEZ_2':'Agro-climatic Resources',
    'GAEZ_3':'Agro-climatic Potential Yield',
    'GAEZ_4':'Suitability and Attainable Yield',
    'GAEZ_5':'Actual Yield and Production',
    'GAEZ_6':'Yield and Production Gap'
}



# Define the columns used in the analysis
GAEZ_columns = {
    "GAEZ_1": ["name", "sub_theme_name", "variable", "year", "model", "rcp", "units", "download_url"],
    "GAEZ_2": ["name", "sub_theme_name", "variable", "year", "model", "rcp", "units", "download_url"],
    "GAEZ_3": ["name", "sub_theme_name", "variable", "year", "model", "rcp", "crop", "water_supply", "units", "input_level", "download_url"],
    "GAEZ_4": ["name", "sub_theme_name", "variable", "year", "model", "rcp", "crop", "water_supply", "units", "input_level", "c02_fertilization", "download_url"],
    "GAEZ_5": ["name", "sub_theme_name", "variable", "year", "crop", "water_supply", "units", "download_url"],
    "GAEZ_6": ["name", "sub_theme_name", "variable", "year", "crop", "water_supply", "units"]
}

# Note
# - GAEZ_4 use attainable yield of <current cropland> because it includes both irrigated and rainfed cropland
# - GAEZ_4 use input_level == "High" because we assume the farming intensity is high in China in the future
GAEZ_filter_con = {
    "GAEZ_4": 'variable == "Average attainable yield of current cropland" and input_level == "High"',
    "GAEZ_5": '(variable == "Yield" or variable == "Harvested area")' 
}

GAEZ_years = {
    "GAEZ_4": ['1981-2010', '2011-2040', '2041-2070','2071-2100'],
    "GAEZ_5": [2010],
}

GAEZ_water_supply = {
    'GAEZ_4':{
        'Irrigation': 'Irrigated', 
        'Rainfed': "Dryland", 
        'Gravity Irrigation': 'Irrigated',
        'Sprinkler Irrigation': 'Irrigated', 
        'Rainfed All Phases ': 'Dryland',  # Note the space at the end
        'Drip Irrigation': 'Irrigated'
    },
    'GAEZ_5':{
        'Irrigated': 'Irrigated',
        'Rainfed': 'Dryland',
        'Rainfed All Phases': 'Dryland',
    }
}

GAEZ_variables ={
    'GAEZ_4': ["year", "model", "rcp", "crop", "water_supply", "c02_fertilization"],
    'GAEZ_5': ["year", "crop", "water_supply", 'variable']
}


GAEZ_year_mid = {
    '2011-2040': 2025,
    '2041-2070': 2055, 
    '2071-2100': 2085
}

Attainable_conversion = {
    'Maize' : 0.87,
    'Wetland rice':  0.875,
    'Wheat': 0.875
}
