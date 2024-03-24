import numpy as np
import rasterio
from rasterio.features import rasterize
import geopandas as gpd

# Read the region shapefile
region_shp = gpd.read_file('data/Vector_boundary/North_china_Plain_Province.shp')
region_shp = region_shp.sort_values('EN_Name').reset_index(drop=True)

# Read the GAEZ_area tif as template
with rasterio.open('data/GAEZ_v4/GAEZ_area_km2.tif') as src:
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

with rasterio.open('data/GAEZ_v4/Province_mask.tif', 'w', **out_meta) as dst:
    # Rasterize the GeoPandas DataFrame
    rasterized = [rasterize([(geom, 1)] , out_shape=out_shape, transform=out_transform, fill=0, all_touched=True, dtype=rasterio.ubyte) 
                  for geom in region_shp.geometry]
    rasterized = np.stack(rasterized, axis=0)
    # Write the rasterized data to the raster file
    dst.write(rasterized.astype(bool))
