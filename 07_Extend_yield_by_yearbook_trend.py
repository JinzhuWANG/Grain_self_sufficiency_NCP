import numpy as np
import xarray as xr
import dask.array as da

from tqdm.auto import tqdm
from scipy.stats import norm, truncnorm
from dask.diagnostics import ProgressBar as Progressbar


# Read data
yearbook_trend = xr.open_dataset('data/results/step_6_yearbook_yield_extrapolated.nc')
yearbook_trend_ratio = yearbook_trend['mean'] / yearbook_trend.sel(year=2020)['mean']
yearbook_trend_ratio = yearbook_trend_ratio

# Read the GAEZ data
GAEZ_yield_mean = xr.open_dataset('data/results/step_5_GAEZ_actual_yield_extended.nc')['data'].sel(year=slice(2020, 2101))
GAEZ_yield_std = xr.open_dataset('data/results/step_3_GAEZ_AY_GYGA_std.nc')['data'].sel(year=slice(2020, 2101))

GAEZ_yield_mean = GAEZ_yield_mean.chunk({'rcp':1,'year': 1, 'y': -1, 'x': -1})
GAEZ_yield_std = GAEZ_yield_std.transpose(*GAEZ_yield_mean.dims).chunk({'year': 1, 'y': -1, 'x': -1})
GAEZ_yield_std = GAEZ_yield_std.where(GAEZ_yield_std > 0, 1e-3)


# Normalize the samples to the range [0, 1]
n_samples = 100
a,b = 0.025, 0.975
loc, scale = 0.5, 0.1
a_transformed, b_transformed = (a - loc) / scale, (b - loc) / scale
samples = truncnorm(a_transformed, b_transformed, loc=loc, scale=scale).rvs(size=n_samples)

# Create a Dask array for the samples and reshape to match the dimensions
samples_da = da.from_array(samples, chunks=(n_samples,))
samples_da = samples_da[:, None, None, None, None, None, None, None, None]  # Adjust dimensions to match GAEZ_yield_std

# Define a function to compute the ppf
def compute_ppf(ppf, mean, std):
    return norm.ppf(ppf, loc=mean, scale=std)

# Apply the function to each block of the Dask arrays using map_blocks
ppf_da = da.map_blocks(
    compute_ppf, 
    samples_da, 
    GAEZ_yield_mean.data[None, ...], 
    GAEZ_yield_std.data[None, ...], 
    dtype=GAEZ_yield_mean.dtype,
    chunks=(n_samples,) + GAEZ_yield_mean.data.chunks
)

with Progressbar():
    ppf_mean = ppf_da.mean(axis=0).compute()



# Define a function to compute the ppf
def compute_ppf(ppf, mean, std):
    return norm.ppf(ppf, loc=mean, scale=std)

# Use map_blocks to apply the function to each block of the dask arrays
ppfs = []
for n in tqdm(samples, total=n_samples):
    ppf_da = da.map_blocks(
        compute_ppf, 
        n, 
        GAEZ_yield_mean, 
        GAEZ_yield_std,
        chunks=GAEZ_yield_mean.chunks
    )
    ppfs.append(ppf_da)

ppfs_da = da.stack(ppfs, axis=0)



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


