import xarray as xr

from glob import glob
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
meta, arr = calculate_area(tif)

arr = xr.DataArray(arr, dims=['y','x'])
arr = arr.rio.write_crs(meta['crs'])
arr = arr.rio.write_transform(meta['transform'])
arr = arr.chunk({'y':BLOCK_SIZE * 100, 'x':BLOCK_SIZE * 100})


with ProgressBar():
    arr.rio.to_raster(output_path, windowed=True, **meta)


