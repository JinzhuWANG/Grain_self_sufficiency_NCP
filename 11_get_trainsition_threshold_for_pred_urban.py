import pandas as pd
import plotnine



# Read data
urban_area_hist = pd.read_csv('data/results/step_8_area_hist_df.csv')
urban_area_pred = pd.read_csv('data/results/step_9_3_2_predict_urban_area_adj_no_decrease.csv')
urban_area_potential = pd.read_csv('data/results/step_8_urban_area_potential.csv')


urban_pred_potential = urban_area_pred.merge(urban_area_potential, on=['Province'])
urban_pred_potential['remain'] = urban_pred_potential.eval('abs(Area_cumsum_km2 - mean)')

idx = urban_pred_potential.groupby(['Province', 'SSP', 'year'])['remain'].idxmin()
potential_threshold = urban_pred_potential.loc[idx].reset_index(drop=True)

# Save to csv
potential_threshold = potential_threshold[['Province', 'SSP', 'year', 'Area_cumsum_km2', 'Potential']]
potential_threshold.to_csv('data/results/step_11_potential_threshold.csv', index=False)




if __name__ == '__main__':
    
    plotnine.options.figure_size = (10, 6)
    plotnine.options.dpi = 100

    g = (plotnine.ggplot()
         + plotnine.geom_point(urban_area_hist, plotnine.aes(x='year', y='Area_cumsum_km2'), color='grey', size=0.05)
         + plotnine.geom_line(potential_threshold, plotnine.aes(x='year', y='Area_cumsum_km2', color='SSP'), size=0.2)
         + plotnine.facet_wrap('~Province')
         + plotnine.theme_bw()
        )
    
    g.save('data/results/fig_step_11_urban_area_prediction.svg')
    



