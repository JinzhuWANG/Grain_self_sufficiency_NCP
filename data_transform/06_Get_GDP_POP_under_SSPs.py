import pandas as pd
import plotnine
import statsmodels.api as sm

from helper_func import  read_ssp_China, read_ssp_NCP, read_yearbook
from helper_func.get_yearbook_records import get_China_GDP, get_China_population



# Read historical data
GDP_China = get_China_GDP()
Population_China = get_China_population()
GDP_NCP = read_yearbook('data/Yearbook/yearbook_GDP_China.csv','GDP')                       # unit: 10k CNY
POP_NCP = read_yearbook('data/Yearbook/yearbook_population_China.csv','population')         # unit: 10k person

# Read prediction data under SSPs
SSP_GDP_China, SSP_Pop_China = read_ssp_China(data_path='data/SSP_China_data')              # unit: (PPP) billion US$2005/yr | million person
GDP_NCP_pred, POP_NCP_pred = read_ssp_NCP()    





# Build a linear regression to transform GDP_Yearbook to GDP_SSP
GDP_China_SSP = pd.merge(GDP_China, SSP_GDP_China, on='year', suffixes=('_Yearbook', '_SSP'))
m = sm.OLS(GDP_China_SSP['GDP_SSP'], sm.add_constant(GDP_China_SSP['GDP_Yearbook'])).fit()


GDP_China['GDP2SSP'] = m.predict(sm.add_constant(GDP_China['GDP']))
GDP_NCP['GDP2SSP'] = m.predict(sm.add_constant(GDP_NCP['Value']))
GDP_NCP_pred['GDP2SSP'] = m.predict(sm.add_constant(GDP_NCP_pred['Value']))


# Save df to disk
GDP_NCP_pred = GDP_NCP_pred.drop(columns=['index','Value', 'type'])
POP_NCP_pred = POP_NCP_pred.drop(columns=['index', 'type'])
GDP_NCP = GDP_NCP.drop(columns=['Value','type'])
POP_NCP = POP_NCP.drop(columns=['type'])

GDP_NCP_pred.to_csv('data/results/GDP_NCP_pred.csv', index=False)
POP_NCP_pred.to_csv('data/results/POP_NCP_pred.csv', index=False)
GDP_NCP.to_csv('data/results/GDP_NCP.csv', index=False)
POP_NCP.to_csv('data/results/POP_NCP.csv', index=False)


# Sanity check
if __name__ == '__main__':
    plotnine.options.figure_size = (12, 8)
    plotnine.options.dpi = 100

    # GDP-China
    g = (plotnine.ggplot()
            + plotnine.geom_point(GDP_China, plotnine.aes(x='year', y='GDP2SSP'))
            + plotnine.geom_line(SSP_GDP_China, plotnine.aes(x='year', y='GDP', color='Scenario'))
            + plotnine.theme_bw()
    )
    
    # GDP-NCP
    g = (plotnine.ggplot()
            + plotnine.geom_point(GDP_NCP, plotnine.aes(x='year', y='GDP2SSP'), size=0.02, color='grey')
            + plotnine.geom_line(GDP_NCP_pred, plotnine.aes(x='year', y='GDP2SSP', color='SSP'))
            + plotnine.facet_wrap('~Province')
            + plotnine.theme_bw()
    )
    
    # Population-China
    g = (plotnine.ggplot()
            + plotnine.geom_point(Population_China, plotnine.aes(x='year', y='Population'), size=0.02, color='grey')
            + plotnine.geom_line(SSP_Pop_China, plotnine.aes(x='year', y='Population', color='Scenario'))
            + plotnine.theme_bw()
    )
    
    # Population-NCP
    g = (plotnine.ggplot()
            + plotnine.geom_point(POP_NCP, plotnine.aes(x='year', y='Value'), size=0.02, color='grey')
            + plotnine.geom_line(POP_NCP_pred, plotnine.aes(x='year', y='Value', color='SSP'))
            + plotnine.facet_wrap('~Province')
            + plotnine.theme_bw()
    )








