import numpy as np
import rasterio
import geopandas as gpd

from rasterio.features import rasterize
from glob import glob

# Read the region shapefile
region_shp = gpd.read_file('data/Vector_boundary/North_china_Plain_Province.shp')
region_shp = region_shp.sort_values('EN_Name').reset_index(drop=True)

# Read one of the raster files to get the shape, transform, and crs
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
