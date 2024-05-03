import pandas as pd
import plotnine
import statsmodels.api as sm

from helper_func import read_ssp, read_yearbook



# Read historical data of China 
GDP_China = pd.read_csv('data/Yearbook/yearbook_GDP_China.csv').T.drop('地区')                                      # unit: 10k CNY
GDP_China = GDP_China.sum(axis=1).reset_index().rename(columns={'index':'year', 0:'GDP'})
GDP_China['year'] = GDP_China['year'].astype('int16')
GDP_China['GDP'] = GDP_China['GDP'].astype('float64')


Population_China = pd.read_excel('data/Yearbook/China_population_1980_2022.xlsx', sheet_name='统计数据')             # unit: 10k person
Population_China = Population_China[Population_China["地区名称"] != '中国']
Population_China = Population_China.groupby(['统计年度'])[['总人口数/万人']].sum().reset_index()
Population_China.columns = ['year','Population']
Population_China['Population'] = Population_China['Population'].astype('float64')
Population_China['Population'] = Population_China['Population']/1e2                                                 # unit: million person


# Read SSP data of China
SSP_GDP_China, SSP_Pop_China = read_ssp(data_path='data/SSP_China_data')                                            # unit: (PPP) billion US$2005/yr | million person



# Read historical data of the North China Plain
GDP_NCP = read_yearbook('data/Yearbook/yearbook_GDP_China.csv','GDP')                                               # unit: 10k CNY
POP_NCP = read_yearbook('data/Yearbook/yearbook_population_China.csv','population')                                 # unit: 10k person



# Read projected SSP data of the North China Plain
GDP_NCP_dfs = []
POP_NCP_dfs = []
for i in range(1,6):
    ncp_gdp = read_yearbook(f"data/SSP_China_data/SSPs_GDP_Prov_v2_SSP{i}.csv", 'GDP')                              # unit: 10k CNY
    ncp_pop = read_yearbook(f"data/SSP_China_data/SSPs_POP_Prov_v2_SSP{i}.csv", 'population')                       # unit: 1 person
    ncp_pop['Value'] = ncp_pop['Value']/1e4                                                                         # unit: 10k person
    ncp_gdp['SSP'] = ncp_pop['SSP'] = f'SSP{i}'
    GDP_NCP_dfs.append(ncp_gdp)
    POP_NCP_dfs.append(ncp_pop)
    
GDP_NCP_pred = pd.concat(GDP_NCP_dfs).reset_index()                                                                # unit: 10k CNY
POP_NCP_pred = pd.concat(POP_NCP_dfs).reset_index()                                                                # unit: 10k person





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








