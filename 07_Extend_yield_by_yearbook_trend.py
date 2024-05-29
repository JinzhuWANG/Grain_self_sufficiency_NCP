import xarray as xr
import plotnine
import rioxarray as rxr

from helper_func import sample_ppf
from helper_func.parameters import UNIQUE_VALUES, Monte_Carlo_num



# Define parameters
chunk_size = {
    'crop': 3,
    'water_supply': 2,
    'y': 160,
    'x': 149,
    'band': 1,
    'year': 1,
    'rcp': 4,
    'c02_fertilization': 2
}



# Read data
yearbook_trend = xr.open_dataset('data/results/step_6_yearbook_yield_extrapolated.nc')

mask = rxr.open_rasterio('data/GAEZ_v4/Province_mask.tif')
mask = [xr.where(mask == idx, 1, 0).expand_dims({'Province': [p]}) for idx,p in enumerate(UNIQUE_VALUES['Province'])]
mask = xr.combine_by_coords(mask).astype('float32') 

GAEZ_yield_mean = xr.open_dataset('data/results/step_5_GAEZ_actual_yield_extended.nc')
GAEZ_yield_mean = GAEZ_yield_mean['data'].sel(year=slice(2020, 2101)).astype('float32').chunk(chunk_size)

GAEZ_yield_std = xr.open_dataset('data/results/step_3_GAEZ_AY_GYGA_std.nc') / 1000   # kg/ha to t/ha
GAEZ_yield_std = GAEZ_yield_std['data'].sel(year=slice(2020, 2101))
GAEZ_yield_std = GAEZ_yield_std.transpose(*GAEZ_yield_mean.dims).astype('float32')
GAEZ_yield_std = GAEZ_yield_std.where(GAEZ_yield_std > 0, 1e-3).chunk(chunk_size)


# Sample from GAEZ mean and std
GAEZ_MC = sample_ppf(GAEZ_yield_mean, GAEZ_yield_std, n_samples=Monte_Carlo_num)
GAEZ_MC = xr.DataArray(
    GAEZ_MC, 
    dims=('sample',) + GAEZ_yield_mean.dims, 
    coords={'sample':range(Monte_Carlo_num), **GAEZ_yield_mean.coords}
    )
GAEZ_MC.name = 'data'


# Sample from yearbook trend
Yearbook_MC = sample_ppf(yearbook_trend['mean'], yearbook_trend['std'], n_samples=Monte_Carlo_num)
Yearbook_MC = xr.DataArray(
    Yearbook_MC, 
    dims=('sample',) + yearbook_trend['mean'].dims, 
    coords={'sample':range(Monte_Carlo_num), **yearbook_trend['mean'].coords}
    )
Yearbook_MC.name = 'data'

# Multiply by the mask, so all pixels inside a province have the same value
Yearbook_MC = Yearbook_MC * mask    






# Save the data
encoding = {'data': {'dtype': 'float32', 'zlib': True}}
GAEZ_MC.to_netcdf('data/results/step_7_GAEZ_actual_MC.nc', encoding=encoding, engine='h5netcdf')













if __name__ == '__main__':
    GAEZ_mean = GAEZ_yield_mean.sel(year=2070)
    GAEZ_MC_mean = GAEZ_MC.mean(dim='sample').sel(year=2070).compute()
    diff = GAEZ_mean - GAEZ_MC_mean

    # Histogram
    plotnine.options.figure_size = (16, 6)
    plotnine.options.dpi = 100
    
    g = (plotnine.ggplot() +
        plotnine.geom_histogram(diff.to_dataframe().query('data > 0'), plotnine.aes(x='data'), bins=30, fill='red', alpha=0.5) +
        plotnine.theme_bw()
        )
    
    # Map
    diff[0,0,...,0,0,0].plot()
   

