import numpy as np
import rasterio
import h5py
import geopandas as gpd

from rasterio.features import rasterize
from glob import glob
from tqdm.auto import tqdm
from affine import Affine
from joblib import Parallel, delayed
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
from itertools import product

from helper_func.parameters import HDF_BLOCK_SIZE




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
out_meta.update({'dtype': rasterio.ubyte, 
                 'count': len(region_shp), 
                 'compress': 'lzw',
                 'nbits':1,
                 'nodata':None})

out_meta_mean = out_meta.copy()
out_meta_mean.update({'dtype': rasterio.float32, 
                 'count': len(region_shp), 
                 'compress': 'lzw',
                 'nbits':1,
                 'nodata':None})

# Rasterize the province shapefile, and 
# 1) update the rasterized data with the src_mask
# 2) divede the mask by pixel count to get the mean_mask
with rasterio.open('data/GAEZ_v4/Province_mask.tif', 'w', **out_meta) as mask_dst, \
     rasterio.open('data/GAEZ_v4/Province_mask_mean.tif', 'w', **out_meta_mean) as mask_dst_mean:
    
    # Rasterize the GeoPandas DataFrame
    rasterized = [rasterize([(geom, 1)] , out_shape=out_shape, transform=out_transform, fill=0, all_touched=False, dtype=rasterio.ubyte) 
                  for geom in region_shp.geometry]
    
    # Update each rasterized province with the src_mask
    mean_masks = []
    for i in range(len(rasterized)):
        rasterized[i] = np.where(src_mask, rasterized[i], 0)
        mean_masks.append(rasterized[i] / np.sum(rasterized[i] > 0))

    # Stack the rasterized data
    rasterized = np.stack(rasterized, axis=0)
    mean_masks = np.stack(mean_masks, axis=0)

    # Write the rasterized data to the raster file
    mask_dst.write(rasterized.astype(bool))
    mask_dst_mean.write(mean_masks)



##############################################
# Creating mask of the LUCC data
##############################################

hdf_ds = h5py.File('data/LUCC/Urban_1990_2019.hdf5', 'r')
hdf_arr = hdf_ds['Array']

hdf_shape = hdf_arr.shape[1:]
hdf_transform = Affine(*hdf_ds['Transform'][:])

process_chunk_size = HDF_BLOCK_SIZE * 20

# Get the top left corners of the windows
windows_starts = list(
    product(
        range(0, hdf_shape[0], process_chunk_size),
        range(0, hdf_shape[1], process_chunk_size),
    )
)


# Create the output dataset
with h5py.File('data/LUCC/LUCC_Province_mask.hdf5', 'w') as mask_ds:
    mask_arr = mask_ds.create_dataset('Array', 
                                    shape=hdf_shape, 
                                    dtype=np.uint8, 
                                    chunks=(HDF_BLOCK_SIZE, HDF_BLOCK_SIZE),
                                    compression='gzip',
                                    compression_opts=9)
    

    for row, col in tqdm(windows_starts, total=len(windows_starts)):
        # Adjust the window size for the final window
        window_height = min(process_chunk_size, hdf_shape[0] - row)
        window_width = min(process_chunk_size, hdf_shape[1] - col)
        window_shape = (window_height, window_width)
        window = rasterio.windows.Window(col, row, window_width, window_height)
        out_transform = rasterio.windows.transform(window, hdf_transform)

        rasterized = [rasterize([(geom, idx) for idx,geom in enumerate(region_shp.geometry)], 
                                out_shape=window_shape, 
                                transform=out_transform, 
                                fill=0, all_touched=False, 
                                dtype=np.uint8)]
        
        
        rasterized = np.array(rasterized).sum(axis=0)

        mask_arr[row:row+window_height, col:col+window_width] = rasterized








