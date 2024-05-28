import numpy as np
import xarray as xr
import dask.array as da

from joblib import Parallel, delayed
from tqdm.auto import tqdm
from scipy.stats import norm, truncnorm
from dask.diagnostics import ProgressBar as Progressbar


# Read data
yearbook_trend = xr.open_dataset('data/results/step_6_yearbook_yield_extrapolated.nc')
yearbook_trend_ratio = yearbook_trend['mean'] / yearbook_trend.sel(year=2020)['mean']
yearbook_trend_ratio = yearbook_trend_ratio

# Read the GAEZ data
chunk_size = {'rcp':-1,'year': 1, 'y': -1, 'x': -1}
GAEZ_yield_mean = xr.open_dataset('data/results/step_5_GAEZ_actual_yield_extended.nc')['data'].sel(year=slice(2020, 2101))
GAEZ_yield_std = xr.open_dataset('data/results/step_3_GAEZ_AY_GYGA_std.nc')['data'].sel(year=slice(2020, 2101))

GAEZ_yield_mean = GAEZ_yield_mean.chunk(chunk_size)
GAEZ_yield_std = GAEZ_yield_std.transpose(*GAEZ_yield_mean.dims).chunk(chunk_size)
GAEZ_yield_std = GAEZ_yield_std.where(GAEZ_yield_std > 0, 1e-3)


# Normalize the samples to the range [0, 1]
n_samples = 100
a,b = 0.025, 0.975
loc, scale = 0.5, 0.1
a_transformed, b_transformed = (a - loc) / scale, (b - loc) / scale
samples = truncnorm(a_transformed, b_transformed, loc=loc, scale=scale).rvs(size=n_samples)


# Define a function to compute the ppf
def compute_ppf(ppfs, mean, std):
    return da.from_array([norm.ppf(p, loc=mean, scale=std) for p in ppfs], chunks=(n_samples,))


ppf_da = da.map_blocks(
        compute_ppf, 
        samples, 
        GAEZ_yield_mean, 
        GAEZ_yield_std, 
        dtype=GAEZ_yield_mean.dtype,
        chunks=(n_samples,) + GAEZ_yield_mean.data.chunks
    )

with Progressbar():
    ppf_mean = ppf_da.mean(axis=0).compute()




# Function to sample percentiles from the mean and std arrays
def sample_from_arr(n, idx=0):
    ppf_da = da.map_blocks(
        compute_ppf, 
        n, 
        GAEZ_yield_mean, 
        GAEZ_yield_std, 
        dtype=GAEZ_yield_mean.dtype,
        chunks=GAEZ_yield_mean.data.chunks
    )[None, ...]
    
    ppf_da = xr.DataArray(
        ppf_da, 
        dims=('sample',) + GAEZ_yield_mean.dims, 
        coords={'sample':[idx], **GAEZ_yield_mean.coords},
        name=None)
    
    ppf_da.name = None 
    
    return ppf_da


task = [delayed(sample_from_arr)(n, idx) for idx, n in enumerate(samples)]
para_obj = Parallel(n_jobs=-1, prefer='threads', return_as='generator')

ppf_xr = []
for res in tqdm(para_obj(task), total=n_samples):
    ppf_xr.append(res)

ppf_xr = xr.combine_by_coords(ppf_xr).chunk({'sample': -1})



with Progressbar():
    ppf_mean = ppf_xr.mean(dim='sample').compute()



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


