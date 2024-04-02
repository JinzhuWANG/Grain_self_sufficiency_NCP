from affine import Affine
from pyproj import CRS
import rasterio
import numpy as np
import h5py
import dask.array as da

from glob import glob

from helper_func.parameters import HDF_BLOCK_SIZE


def haversine(coord1, coord2):
    """
    Calculate the great circle distance in kilometers between two points 
    on the earth (specified in decimal degrees, with coordinates in the order of longitude, latitude).
    
    Arguments:
    coord1 -- tuple containing the longitude and latitude of the first point (lon1, lat1)
    coord2 -- tuple containing the longitude and latitude of the second point (lon2, lat2)
    
    Returns:
    distance in kilometers.
    """
    # Extract longitude and latitude, then convert from decimal degrees to radians
    lon1, lat1 = da.radians(coord1)
    lon2, lat2 = da.radians(coord2)
    
    # Haversine formula 
    dlat = abs(lat2 - lat1) 
    dlon = abs(lon2 - lon1) 
    
    a = da.sin(dlat/2)**2 + da.cos(lat1) * da.cos(lat2) * da.sin(dlon/2)**2
    c = 2 * da.arcsin(da.sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r



def get_meta(tif:str):

    # If the tif end with .tif/tiff, then read the tif file
    if tif.endswith('.tif') or tif.endswith('.tiff'):
        with rasterio.open(tif) as src:
            meta = src.meta
    elif tif.endswith('.hdf5'):
        meta = {'driver': 'GTiff',
                'dtype': None,
                'nodata': None,
                'count': 1,
                'crs': CRS.from_epsg(4326),
                'transform': None}
        with h5py.File(tif, 'r') as f:
            meta['width'] = f['Array'].shape[2]
            meta['height'] = f['Array'].shape[1]
            meta['dtype'] = f['Array'].dtype
            meta['transform'] = Affine(*f['Transform'])
    return meta


def calculate_area(tif:str, output_path:str):

    # Get the meta information of the raster
    meta = get_meta(tif)
        
    # Check if the raster is in a geographic coordinate system
    if not meta['crs'].is_geographic:
        raise ValueError("The raster is not in a geographic coordinate system!")
    
    # Get raster size
    width, height = meta['width'], meta['height']
    # Get raster's geotransform
    transform = meta['transform']
    
    # Get source metadata, and update it to match the output raster
    meta.update(compress='lzw',
                dtype=rasterio.float32,
                count=1,
                nodata=None)
    
    # Generate pixel coordinates
    rows, cols = da.arange(height, chunks=1024*10), da.arange(width,chunks=1024*10)
    row_coords, col_coords = da.meshgrid(rows, cols, indexing='ij')

    # Apply geotransform to get geographical coordinates
    # Transform (col, row) -> (x, y)
    x_coords, y_coords = transform * (col_coords, row_coords)
    xy_coords = da.stack((x_coords, y_coords), axis=0)

    x_coords_right, y_coords_right = transform * (col_coords + 1, row_coords)
    xy_coords_right = da.stack((x_coords_right, y_coords_right), axis=0)
    
    x_coords_bottom, y_coords_bottom = transform * (col_coords, row_coords + 1)
    xy_coords_bottom = da.stack((x_coords_bottom, y_coords_bottom), axis=0)

    # Calculate the area of each pixel
    length_right = haversine(xy_coords, xy_coords_right)
    length_bottom = haversine(xy_coords, xy_coords_bottom)
    area_arry = length_right * length_bottom

    # Save the area array to the output raster
    with rasterio.open(output_path,'w', **meta) as dst:
        for i in range(0, height, HDF_BLOCK_SIZE):
            for j in range(0, width, HDF_BLOCK_SIZE):
                chunk = area_arry[i:i+HDF_BLOCK_SIZE, j:j+HDF_BLOCK_SIZE].compute()
                dst.write(chunk, 1, window=rasterio.windows.Window(j, i, chunk.shape[1], chunk.shape[0]))
        
        print(f"Area of {tif} calculated and saved to {output_path}")


    
if __name__ == '__main__':

    # Calculate GAEZ_tif area
    tif = glob('data/GAEZ_v4/GAEZ_tifs/*tif')[0]
    output_path = 'data/GAEZ_v4/GAEZ_area_km2.tif'
    calculate_area(tif, output_path)

    # Calculate LUCC_tif area
    hdf = 'data/LUCC/Urban_1990_2019.hdf5'
    output_path = 'data/LUCC/LUCC_area_km2.tif'
    calculate_area(hdf, output_path)
        



