import dask
import numpy as np
import xarray as xr
import dask
import dask.array as da

from dask.diagnostics import ProgressBar as Progressbar



# Read data
yearbook_trend = xr.open_dataset('data/results/step_6_yearbook_yield_extrapolated.nc')
yearbook_trend_ratio = yearbook_trend / yearbook_trend.sel(year=2020)['mean']
yearbook_trend_ratio = yearbook_trend_ratio.astype(np.float32)

# Read the GAEZ data
GAEZ_yield_mean = xr.open_dataset('data/results/step_5_GAEZ_actual_yield_extended.nc', chunks='auto')['data'].astype(np.float32)
GAEZ_yield_std = xr.open_dataset('data/results/step_3_GAEZ_AY_GYGA_std.nc', chunks='auto')['data'].sel(year=slice(2020, 2101)).astype(np.float32)

def generate_samples(mean, std, size):
    return da.random.normal(mean, std, size)

GAEZ_MC = xr.apply_ufunc(generate_samples, 
                         GAEZ_yield_mean, 
                         GAEZ_yield_std.where(GAEZ_yield_std > 0, 0),
                         1000,
                         input_core_dims=[[], [], []],
                         output_core_dims=[['sample']],
                         vectorize=True,
                         dask='parallelized',
                         dask_gufunc_kwargs={'output_sizes': {'sample': 1000}},).astype(np.float32)

# Adjust the yield by the yearbook trend
GAEZ_MC_adj_yb = GAEZ_MC * yearbook_trend_ratio['mean']
GAEZ_MC_adj_yb = GAEZ_MC_adj_yb.astype(np.float32)

# Get the uncertainty range from the yearbook trend
yearbook_range = xr.DataArray(
    da.random.normal(0, 1, (1000,) + yearbook_trend_ratio['std'].shape),
    dims=('sample',) + yearbook_trend_ratio['std'].dims,
    coords={'sample': np.arange(1000), **yearbook_trend_ratio['std'].coords}
) * yearbook_trend_ratio['std'].where(yearbook_trend_ratio['std'] > 0, 0)

yearbook_range = yearbook_range.astype(np.float32)

# Add the yearbook trend uncertainty range to the yield
GAEZ_MC_adj_yb = GAEZ_MC_adj_yb + yearbook_range

# Rechunk data to ~100MB chunks
chunk_size = {i:1 for i in GAEZ_MC_adj_yb.dims}
chunk_size.update({'sample': 1000, 'y': 100, 'x': -1})
GAEZ_MC_adj_yb = GAEZ_MC_adj_yb.chunk(chunk_size)

with Progressbar():
    mean = GAEZ_MC_adj_yb.mean(dim='sample').compute()
    std = GAEZ_MC_adj_yb.std(dim='sample').compute()

encoding = {'data': {'zlib': True, 'complevel': 9, 'dtype': 'float32'}}
mean.to_netcdf('data/results/step_7_GAEZ_MC_adj_yb_mean.nc', encoding=encoding)
std.to_netcdf('data/results/step_7_GAEZ_MC_adj_yb_std.nc', encoding=encoding)


