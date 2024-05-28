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
chunk_size = {
    'crop': 1,
    'water_supply': 2,
    'y': 160,
    'x': 149,
    'band': 1,
    'year': 2,
    'rcp': 2,
    'c02_fertilization': 2
}

GAEZ_yield_mean = xr.open_dataset('data/results/step_5_GAEZ_actual_yield_extended.nc')
GAEZ_yield_mean = GAEZ_yield_mean['data'].sel(year=slice(2020, 2101)).astype('float32').chunk(chunk_size)

GAEZ_yield_std = xr.open_dataset('data/results/step_3_GAEZ_AY_GYGA_std.nc')['data'].sel(year=slice(2020, 2101))
GAEZ_yield_std = GAEZ_yield_std.transpose(*GAEZ_yield_mean.dims).astype('float32')
GAEZ_yield_std = GAEZ_yield_std.where(GAEZ_yield_std > 0, 1e-3).chunk(chunk_size)


# Normalize the samples to the range [0, 1]
n_samples = 100
a,b = 0.025, 0.975
loc, scale = 0.5, 0.1
a_transformed, b_transformed = (a - loc) / scale, (b - loc) / scale
samples = truncnorm(a_transformed, b_transformed, loc=loc, scale=scale).rvs(size=n_samples)


def compute_ppf(p, mean, std):
    return norm.ppf(p, loc=mean, scale=std) 


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


