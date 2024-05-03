import pandas as pd

from helper_func import read_yearbook
from helper_func.parameters import UNIQUE_VALUES


# Read the historical data
urban_area_hist = pd.read_csv('data/results/area_hist_df.csv')          # unit: km2

GDP_NCP_hist = pd.read_csv('data/results/GDP_NCP.csv')                  # unit: (PPP) billion US$2005/yr                                        
POP_NCP_hist = pd.read_csv('data/results/POP_NCP.csv')                  # unit: 10k person                          


# Read the pred GDP and population data
GDP_NCP_pred = pd.read_csv('data/results/GDP_NCP_pred.csv')             # unit: (PPP) billion US$2005/yr
POP_NCP_pred = pd.read_csv('data/results/POP_NCP_pred.csv')             # unit: 10k person



# Modeling the urban_area ~ GDP + Population
area_GDP_POP = urban_area_hist.merge(GDP_NCP_hist, on=['year','Province']).merge(POP_NCP_hist, on=['year','Province'])



