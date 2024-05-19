import rasterio
import xarray as xr
import numpy as np
import rasterio

from glob import glob
from tqdm.auto import tqdm
from dask.diagnostics import ProgressBar
from helper_func.calculate_WGS_pix_area_km2 import calculate_area
from helper_func.parameters import BLOCK_SIZE



# Save the GAEZ_tif area
tif = glob('data/GAEZ_v4/GAEZ_tifs/*tif')[0]
output_path = 'data/GAEZ_v4/GAEZ_area_km2.tif'
meta, arr = calculate_area(tif)

arr = xr.DataArray(arr,dims=['y','x'])
arr = arr.rio.write_crs(meta['crs'])
arr = arr.rio.write_transform(meta['transform'])


# Save the LUCC_tif area
tif = 'data/LUCC/Norm_Urban_1990_2019.tif'
output_path = 'data/LUCC/LUCC_area_km2.tif'
work_size = BLOCK_SIZE * 30
meta, arr = calculate_area(tif, chunk_size=work_size)

arr = xr.DataArray(arr, dims=['y','x'])
arr = arr.rio.write_crs(meta['crs'])
arr = arr.rio.write_transform(meta['transform'])

# Get the chunks
chunk_sizes = arr.chunks
chunk_num = np.prod([len(i) for i in chunk_sizes])

with rasterio.open(output_path, 'w', **meta) as dst:
    pbar = tqdm(total=chunk_num)
    # Loop through the chunks
    for y_chunk, y_start in zip(chunk_sizes[0], np.cumsum([0] + list(chunk_sizes[0][:-1]))):
        for x_chunk, x_start in zip(chunk_sizes[1], np.cumsum([0] + list(chunk_sizes[1][:-1]))):
            # Get the chunk data
            chunk = arr.isel(y=slice(y_start, y_start + y_chunk), x=slice(x_start, x_start + x_chunk)).data
            dst.write(chunk, 1, window=((y_start, y_start + y_chunk), (x_start, x_start + x_chunk)))
            pbar.update(1)




