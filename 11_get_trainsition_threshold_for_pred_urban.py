import pandas as pd
import plotnine



# Read data
urban_area_hist = pd.read_csv('data/results/step_8_area_hist_df.csv')
urban_area_pred = pd.read_csv('data/results/step_9_Urban_SSP_pred_Gao_diff.csv')
urban_area_potential = pd.read_csv('data/results/step_8_urban_area_potential.csv')


urban_pred_potential = urban_area_pred.merge(urban_area_potential, on=['Province'])
urban_pred_potential['remain'] = urban_pred_potential.eval('abs(Area_cumsum_km2 - area_km2_adj)')

idx = urban_pred_potential.groupby(['Province', 'ssp', 'year'])['remain'].idxmin()
potential_threshold = urban_pred_potential.loc[idx].reset_index(drop=True)
potential_threshold.query('ssp == "SSP5" & Province == "Anhui"')

# Save to csv
potential_threshold = potential_threshold[['Province', 'ssp', 'year', 'Area_cumsum_km2', 'Potential']]
potential_threshold.to_csv('data/results/step_11_potential_threshold.csv', index=False)




if __name__ == '__main__':
    
    plotnine.options.figure_size = (10, 6)
    plotnine.options.dpi = 100

    g = (plotnine.ggplot()
         + plotnine.geom_line(potential_threshold, plotnine.aes(x='year', y='Area_cumsum_km2', color='ssp'), size=0.2)
         + plotnine.facet_wrap('~Province')
         + plotnine.theme_bw()
        )
    
    g.save('data/results/fig_step_11_urban_area_prediction_with_Gao_SSP.svg')
    



