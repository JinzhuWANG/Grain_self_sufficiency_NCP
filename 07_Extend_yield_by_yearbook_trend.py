import numpy as np
import xarray as xr
import dask.array as da

from statistics import NormalDist
from dask.diagnostics import ProgressBar as Progressbar

sample_size = 100

# Read data
yearbook_trend = xr.open_dataset('data/results/step_6_yearbook_yield_extrapolated.nc')
yearbook_trend_ratio = yearbook_trend / yearbook_trend.sel(year=2020)['mean']
yearbook_trend_ratio = yearbook_trend_ratio

# Read the GAEZ data
GAEZ_yield_mean = xr.open_dataset('data/results/step_5_GAEZ_actual_yield_extended.nc', chunks='auto')['data']
GAEZ_yield_std = xr.open_dataset('data/results/step_3_GAEZ_AY_GYGA_std.nc', chunks='auto')['data'].sel(year=slice(2020, 2101))




GAEZ_MC = xr.DataArray(
    NormalDist(mu=GAEZ_yield_mean.values, sigma=GAEZ_yield_std.values).inv_cdf(0.95),
    dims=('sample',) + GAEZ_yield_std.dims,
    coords={'sample': np.arange(sample_size), **GAEZ_yield_std.coords}
) 










# Adjust the yield by the yearbook trend
GAEZ_MC_adj_yb = GAEZ_MC * yearbook_trend_ratio['mean']


# Get the uncertainty range from the yearbook trend
yearbook_range = xr.DataArray(
    da.random.normal(0, 1, (sample_size,) + yearbook_trend_ratio['std'].shape),
    dims=('sample',) + yearbook_trend_ratio['std'].dims,
    coords={'sample': np.arange(sample_size), **yearbook_trend_ratio['std'].coords}
) * yearbook_trend_ratio['std'].where(yearbook_trend_ratio['std'] > 0, 0)

yearbook_range = yearbook_range.astype(np.float32)

# Add the yearbook trend uncertainty range to the yield
GAEZ_MC_adj_yb = GAEZ_MC_adj_yb + yearbook_range

# Rechunk data to ~100MB chunks
chunk_size = {i:1 for i in GAEZ_MC_adj_yb.dims}
chunk_size.update({'sample': sample_size, 'y': 100, 'x': -1})
GAEZ_MC_adj_yb = GAEZ_MC_adj_yb.chunk(chunk_size)

with Progressbar():
    mean = GAEZ_MC_adj_yb.mean(dim='sample').compute()
    std = GAEZ_MC_adj_yb.std(dim='sample').compute()


mean.name = 'data'
std.name = 'data'
encoding = {'data': {'zlib': True, 'complevel': 9, 'dtype': 'float32'}}
mean.to_netcdf('data/results/step_7_GAEZ_MC_adj_yb_mean.nc', encoding=encoding)
std.to_netcdf('data/results/step_7_GAEZ_MC_adj_yb_std.nc', encoding=encoding)


