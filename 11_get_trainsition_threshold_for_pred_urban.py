import pandas as pd



# Read data
urban_area_hist = pd.read_csv('data/results/area_hist_df.csv')
urban_area_pred = pd.read_csv('data/results/Urban_SSP_pred_Gao_diff.csv')
urban_area_potential = pd.read_csv('data/results/urban_area_potential.csv')


urban_pred_potential = urban_area_pred.merge(urban_area_potential, on='Province')









