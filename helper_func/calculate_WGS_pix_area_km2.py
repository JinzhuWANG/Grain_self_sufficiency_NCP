
import os
import rasterio
import numpy as np

from glob import glob
from sys import argv


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
    lon1, lat1 = np.radians(coord1)
    lon2, lat2 = np.radians(coord2)
    
    # Haversine formula 
    dlat = abs(lat2 - lat1) 
    dlon = abs(lon2 - lon1) 
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r


# Read geotiff file in data folder
def calculate_area(tif:str, output_path:str):

    # Open the source raster
    with rasterio.open(tif) as src:
        
        # Check if the raster is in a geographic coordinate system
        if not src.crs.is_geographic:
            raise ValueError("The raster is not in a geographic coordinate system!")
        
        # Get raster size
        width, height = src.width, src.height
        # Get raster's geotransform
        transform = src.transform
        
        # Get source metadata, and update it to match the output raster
        meta = src.meta.copy()
        meta.update(compress='lzw',
                    dtype=rasterio.float32,
                    count=1)
        
        
        # Generate pixel coordinates
        rows, cols = np.arange(height), np.arange(width)
        row_coords, col_coords = np.meshgrid(rows, cols, indexing='ij')
        
        # Apply geotransform to get geographical coordinates
        # Transform (col, row) -> (x, y)
        x_coords, y_coords = transform * (col_coords, row_coords)
        xy_coords = np.stack((x_coords, y_coords), axis=0)
        
        x_coords_right, y_coords_right = transform * (col_coords + 1, row_coords)
        xy_coords_right = np.stack((x_coords_right, y_coords_right), axis=0)
        
        x_coords_bottom, y_coords_bottom = transform * (col_coords, row_coords + 1)
        xy_coords_bottom = np.stack((x_coords_bottom, y_coords_bottom), axis=0)
    
        
        
        # Calculate the area of each pixel
        length_right = haversine(xy_coords, xy_coords_right)
        length_bottom = haversine(xy_coords, xy_coords_bottom)
        area_arry = length_right * length_bottom
        
        # Save the area array to the output raster
        with rasterio.open(output_path,'w', **meta) as dst:
            dst.write(area_arry, 1)
            
            print(f"Area of {tif} calculated and saved to {output_path}")


    
    
if __name__ == '__main__':
    # Get the first GAEZ_tif file
    tif = glob('data/GAEZ_v4/GAEZ_tifs/*tif')[0]
    
    # Get the output path
    output_path = 'data/GAEZ_v4/GAEZ_area_km2.tif'
    
    calculate_area(tif, output_path)
        



