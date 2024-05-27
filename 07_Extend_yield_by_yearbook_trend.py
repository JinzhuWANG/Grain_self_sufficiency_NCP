import numpy as np
import xarray as xr
import dask.array as da

from scipy.stats import norm, truncnorm
from dask.diagnostics import ProgressBar as Progressbar


# Read data
yearbook_trend = xr.open_dataset('data/results/step_6_yearbook_yield_extrapolated.nc')
yearbook_trend_ratio = yearbook_trend['mean'] / yearbook_trend.sel(year=2020)['mean']
yearbook_trend_ratio = yearbook_trend_ratio

# Read the GAEZ data
GAEZ_yield_mean = xr.open_dataset('data/results/step_5_GAEZ_actual_yield_extended.nc', chunks='auto')['data']
GAEZ_yield_std = xr.open_dataset('data/results/step_3_GAEZ_AY_GYGA_std.nc', chunks='auto')['data'].sel(year=slice(2020, 2101))
GAEZ_yield_std = GAEZ_yield_std.transpose(*GAEZ_yield_mean.dims)


chunk_size = {i:1 for i in GAEZ_yield_std.dims}



n_sample = 100
samples = np.random.normal(0, 1, n_sample)
min_sample, max_sample = np.min(samples) + 1e-5, np.max(samples) + 1e-5

# Normalize the samples to the range [0, 1]
normalized_samples = (samples - min_sample) / (max_sample - np.min(samples))
normalized_samples = da.from_array(normalized_samples, chunks='auto')


# Define a function to compute the ppf
def compute_ppf(ppf, mean, std):
    return norm.ppf(ppf, loc=mean, scale=std)

# Use map_blocks to apply the function to each block of the dask arrays
ppf_da = da.map_blocks(compute_ppf, 
                       normalized_samples.reshape([-1] + [1]*(GAEZ_yield_mean.ndim)), 
                       GAEZ_yield_mean, 
                       GAEZ_yield_std)



# Convert the dask array back to an xarray DataArray
GAEZ_MC = xr.DataArray(
    ppf_da,
    dims=('sample',) + GAEZ_yield_std.dims,
    coords={'sample': np.arange(normalized_samples.shape[0]), **GAEZ_yield_std.coords}
)



# Adjust the yield by the yearbook trend
GAEZ_MC_adj_yb = GAEZ_MC * yearbook_trend_ratio['mean']

# Generate the Monte Carlo simulation
GAEZ_adj_MC = xr.DataArray(
    norm.ppf(0.5, loc=GAEZ_MC_adj_yb, scale=GAEZ_yield_std),
    dims=GAEZ_MC_adj_yb.dims,
    coords=GAEZ_MC_adj_yb.coords
)

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


