import pandas as pd


# Read the GAEZ_df which records the metadata of the GAEZ data
GAEZ_df = pd.read_csv('data/GAEZ_v4/GAEZ_df.csv')



GAEZ_df.query('year')
