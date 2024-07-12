from matplotlib import scale
import pandas as pd
import plotnine

from scipy.interpolate import interp1d
from helper_func import read_yearbook
from helper_func.get_yearbook_records import get_NCP_urban_population_ratio
from helper_func.parameters import UNIQUE_VALUES


def extapolate_area(df):
    yr = df['year']
    area = df['Area_cumsum_km2']
    f = interp1d(yr, area, kind='linear', fill_value='extrapolate')
    return f(range(1990,2021))


# Read the historical cumulative urban area
population_hist = read_yearbook('data/Yearbook/yearbook_population_China.csv','population')         # unit: 10k person
population_hist['Population (million)'] = population_hist['Value'] / 100
population_hist = population_hist.query('year >= 1990')

population_ssp = pd.read_csv('data/results/POP_NCP_pred.csv')
population_ssp['Population (million)'] = population_ssp['Value'] / 100

urban_area_hist = pd.read_csv('data/results/step_8_area_hist_df.csv')                               # unit: km2
urban_area_hist_ext = pd.DataFrame()
for p, df in urban_area_hist.groupby('Province'):
    df_ext = pd.DataFrame({'year':range(1990,2021)})
    df_ext['Province'] = p
    df_ext['Area_cumsum_km2'] = extapolate_area(df)
    urban_area_hist_ext = pd.concat([urban_area_hist_ext, df_ext], axis=0)




# Merge the urban area with the urban population ratio
urban_population_ratio = get_NCP_urban_population_ratio()
urban_area_pop_ratio = urban_area_hist_ext.merge(urban_population_ratio, on=['Province', 'year'])



# Normalize the population and urban-population ratio
total_pop_and_urban_ratio = population_hist.merge(
    urban_area_pop_ratio, 
    on=['Province', 'year'])[['Province', 'year', 'Population (million)', 'urban_pop_ratio']]

total_pop_and_urban_ratio = total_pop_and_urban_ratio.sort_values(
    by=['Province', 'year']
).reindex()



# Save the results
urban_area_hist_ext.to_csv('data/results/step_9_1_1_urban_area_ext.csv', index=False)
total_pop_and_urban_ratio.to_csv('data/results/step_9_1_2_total_pop_and_urban_ratio.csv', index=False)


if __name__ == '__main__':
    plotnine.options.figure_size = (8, 6)
    plotnine.options.dpi = 100
    
    
    # Plot the population
    g = (plotnine.ggplot() +
            plotnine.geom_point(
                population_hist, 
                plotnine.aes(x='year', y='Population (million)'), color='grey', size=0.03) +
            plotnine.facet_wrap('~Province', scales='free') +
            plotnine.theme_bw() 
            )
    g.save('data/results/fig_step_9_1_0_total_population_hist.svg')
    
    g = (plotnine.ggplot() +
            plotnine.geom_point(
                population_hist, 
                plotnine.aes(x='year', y='Population (million)'), color='grey', size=0.03) +
            # plotnine.geom_line(
            #     population_ssp, 
            #     plotnine.aes(x='year', y='Population (million)', color='SSP')) +
            plotnine.facet_wrap('~Province', scales='free') +
            plotnine.theme_bw() 
            )
    g.save('data/results/fig_step_9_1_1_total_population_hist_and_future.svg')
    
    # Plot the extrapolated urban area
    g = (plotnine.ggplot() +
         plotnine.geom_line(
             urban_area_hist_ext, 
             plotnine.aes(x='year', y='Area_cumsum_km2', color='Province')) +
         plotnine.geom_point(
             urban_area_hist,
             plotnine.aes(x='year', y='Area_cumsum_km2', color='Province')) +
        plotnine.theme_bw() 
        )
    
    g.save('data/results/fig_step_9_1_2_historic_urban_area_km2.svg')
    
    # Plot the urban population ratio
    g = (plotnine.ggplot() +
         plotnine.geom_line(
             urban_area_pop_ratio, 
             plotnine.aes(x='year', y='urban_pop_ratio', color='Province')) +
        plotnine.theme_bw() 
        )
    g.save('data/results/fig_step_9_1_3_urban_population_ratio.svg')
    
    # Plot the population v.s. urban-population ratio
    g = (plotnine.ggplot() +
        plotnine.geom_point(
            total_pop_and_urban_ratio, 
            plotnine.aes(
                x='Population (million)', 
                y='urban_pop_ratio', 
                color='Province')) +
        plotnine.theme_bw() +
        plotnine.facet_wrap('~Province', scales='free')
        )
    
    g.save('data/results/fig_step_9_1_4_total_population_v.s._urban_population_ratio.svg')