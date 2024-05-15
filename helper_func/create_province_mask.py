import chunk
import numpy as np
import dask.array as da
import xarray as xr
import rasterio
import rioxarray
import h5py
import geopandas as gpd

from rasterio.features import rasterize
from glob import glob
from tqdm.auto import tqdm
from affine import Affine
from itertools import product

from helper_func.parameters import HDF_BLOCK_SIZE

work_size = HDF_BLOCK_SIZE * 100


# Read the region shapefile
region_shp = gpd.read_file('data/Vector_boundary/North_china_Plain_Province.shp')
region_shp = region_shp.sort_values('EN_Name').reset_index(drop=True)



##############################################
# Creating mask of the GAEZ data
##############################################
GAEZ_tif = glob('data/GAEZ_v4/GAEZ_tifs/*.tif')[0]
with rasterio.open(GAEZ_tif) as src:
    src_mask = src.read(1) > 0 # Only the pixels with values > 0 are valid
    out_shape = src.shape
    out_transform = src.transform
    out_crs = src.crs
    out_meta = src.meta.copy()

# Save the province mask
out_meta.update({'dtype': rasterio.int8, 
                 'count': 1, 
                 'compress': 'lzw',
                 'nodata':-1})

out_meta_mean = out_meta.copy()
out_meta_mean.update({'dtype': rasterio.float32, 
                      'nodata':None})

# Rasterize the province shapefile, and 
# 1) update the rasterized data with the src_mask
# 2) divede the mask by pixel count to get the mean_mask
with rasterio.open('data/GAEZ_v4/Province_mask.tif', 'w', **out_meta) as mask_dst, \
     rasterio.open('data/GAEZ_v4/Province_mask_mean.tif', 'w', **out_meta_mean) as mask_dst_mean:
    
    # Rasterize the GeoPandas DataFrame
    rasterized = [rasterize([(row.geometry, idx + 1)], out_shape=out_shape, transform=out_transform, fill=0, all_touched=False, dtype=rasterio.int8) 
                  for idx,row in region_shp.iterrows()]
    
    mask_sum = np.array(rasterized).sum(axis=0)
    mask_sum = mask_sum - 1
    mask_sum = np.where(mask_sum > len(region_shp) - 1, len(region_shp) - 1, mask_sum)
    
    # Update each rasterized province with the src_mask
    mean_masks = []
    for mask in rasterized:
        mask = mask * src_mask > 0
        mean = mask / np.nansum(mask)
        mean_masks.append(mean.astype(np.float32))
    
    mask_mean = np.array(mean_masks).sum(axis=0)


    # Write the rasterized data to the first band of the raster file
    mask_dst.write(mask_sum.astype(np.int8), 1)
    mask_dst_mean.write(mask_mean.astype(np.float32), 1)



##############################################
# Creating mask of the LUCC data
##############################################

lucc_xr = rioxarray.open_rasterio('data/LUCC/Norm_Urban_1990_2019.tif', 
                                  chunks={'x': work_size, 'y': work_size})


def rasterize_chunk(da_block, gdf):
    """Rasterize a chunk of data."""
    # Get the affine transform for the current chunk
    transform = da_block.rio.transform()

    # Define the shapes and values to rasterize
    shapes = [(row.geometry, idx + 1 ) for idx, row in gdf.iterrows()]

    # Rasterize the shapes onto the chunk
    rasterized = rasterize(shapes, out_shape=(len(da_block.y), len(da_block.x)), fill=0, transform=transform, dtype=np.float32)
    rasterized = rasterized - 1
    rasterized = np.expand_dims(rasterized, axis=0)

    return xr.DataArray(rasterized, coords=da_block.coords, dims=da_block.dims)


# Apply the rasterization function using map_blocks
result = lucc_xr.map_blocks(rasterize_chunk, kwargs={'gdf': region_shp}, template=lucc_xr)
result.rio.to_raster('data/LUCC/LUCC_Province_mask.tif', chunks={ 'x': HDF_BLOCK_SIZE, 'y': HDF_BLOCK_SIZE}, dtype=np.int8, compress='lzw')









