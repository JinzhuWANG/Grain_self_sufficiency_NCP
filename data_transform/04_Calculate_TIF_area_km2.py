import rasterio
import xarray as xr
import rioxarray as rxr
import numpy as np
import rasterio

from glob import glob
from tqdm.auto import tqdm
from dask.diagnostics import ProgressBar
from helper_func.calculate_WGS_pix_area_km2 import calculate_area
from helper_func.parameters import BLOCK_SIZE



# Save the GAEZ_tif area
tif = glob('data/GAEZ_v4/GAEZ_tifs/*tif')[0]
output_path = 'data/GAEZ_v4/GAEZ_area_km2.nc'
calculate_area(path=tif, output_path=output_path)



# Save the LUCC_tif area
path = 'data/LUCC/Norm_Urban_1990_2019.nc'
output_path = 'data/LUCC/LUCC_area_km2.nc'
work_size = BLOCK_SIZE * 30
calculate_area(path, chunk_size=work_size, output_path=output_path)



