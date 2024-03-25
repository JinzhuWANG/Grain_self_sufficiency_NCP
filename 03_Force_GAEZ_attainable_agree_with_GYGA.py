import pandas as pd

# Read the GYGA attainable yield data
GYGA_PY = pd.read_csv('data/GYGA/GYGA_attainable_filled.csv')
GYGA_PY_2010 = GYGA_PY.groupby(['Province','crop','water_supply']).mean(numeric_only=True).reset_index()


# Read the GAEZ attainable yield data
GAEZ_PY = pd.read_pickle('data/results/GAEZ_attainable.pkl')
GAEZ_PY_2010 = GAEZ_PY['mean'].apply(lambda x: x[0]).tolist()

