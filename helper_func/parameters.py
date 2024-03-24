# Define the  crops and GAEZ used in the analysis
crops = ['Maize','Wetland rice','Wheat']

GAEZ_input = ['GAEZ_4',
              'GAEZ_5']


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


GAEZ_variables = {
    "GAEZ_4": ["Average attainable yield of current cropland"],
    "GAEZ_5": ["Yield"],
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

