import xarray as xr
import rioxarray as rxr
import dask.array as da


from helper_func.parameters import BLOCK_SIZE


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



def calculate_area(path:str, chunk_size:int=BLOCK_SIZE, output_path:str=None):
    """
    Calculate the area of each pixel in a raster dataset.

    Parameters:
    path (str): The path to the raster dataset.
    chunk_size (int, optional): The chunk size for processing the raster dataset. Defaults to BLOCK_SIZE.
    output_path (str, optional): The path to save the output raster dataset. Defaults to None.

    Returns:
    None
    """

    # Get the meta information of the raster
    if path.endswith('.nc'):
        src = xr.open_dataset(path)
    elif path.endswith('.tif') or path.endswith('.tiff'):
        src = rxr.open_rasterio(path)
        src = xr.Dataset({'data': src})
    else:
        raise ValueError('Only (.nc) and (.tif/.tiff) files are supported.')

    width, height, transform = src.rio.width, src.rio.height, src.rio.transform()

    # Generate pixel coordinates
    rows, cols = da.arange(height, chunks=chunk_size), da.arange(width,chunks=chunk_size)
    row_coords, col_coords = da.meshgrid(rows, cols, indexing='ij')

    # Calculate col_coords + 1 and row_coords + 1 once
    col_coords_plus_one = col_coords + 1
    row_coords_plus_one = row_coords + 1

    # Apply geotransform to get geographical coordinates
    # Transform (col, row) -> (x, y)
    x_coords, y_coords = transform * (col_coords, row_coords)
    xy_coords = da.stack((x_coords, y_coords), axis=0)

    # Reuse col_coords_plus_one and row_coords_plus_one
    x_coords_right, y_coords_right = transform * (col_coords_plus_one, row_coords)
    xy_coords_right = da.stack((x_coords_right, y_coords_right), axis=0)

    x_coords_bottom, y_coords_bottom = transform * (col_coords, row_coords_plus_one)
    xy_coords_bottom = da.stack((x_coords_bottom, y_coords_bottom), axis=0)

    # Calculate the area of each pixel
    length_right = haversine(xy_coords, xy_coords_right)
    length_bottom = haversine(xy_coords, xy_coords_bottom)
    
    area_arry = length_right * length_bottom
    area_arry = area_arry.astype('float32')[None,...]

    src['data'].values = area_arry
    encoding = {'data': {'zlib': True, 'dtype': 'float32', 'complevel': 9, 'chunksizes': (1, BLOCK_SIZE, BLOCK_SIZE)}}
    src.to_netcdf(output_path, encoding=encoding, engine='h5netcdf')

        



