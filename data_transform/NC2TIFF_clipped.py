import os
import h5py
import dask.array as da
import rasterio
import rioxarray
import geopandas as gpd

from glob import glob
from tqdm.auto import tqdm
from rasterio.errors import CRSError

from helper_func.parameters import HDF_BLOCK_SIZE



# Read the pix area data
ssp_urban_pix_hdf = h5py.File('data/Urban_1km_1km/clipped/Urban_area_km2.hdf5', 'r') 
ssp_urban_pix_km2 = da.from_array(ssp_urban_pix_hdf['area'], chunks=(HDF_BLOCK_SIZE, HDF_BLOCK_SIZE))


# Function to clip the urban data
def clip_urban(in_path:str, out_path:str, region_shp:gpd.GeoDataFrame):
        
    # Get the chunk size
    with h5py.File(in_path, 'r') as in_hdf:
        k = list(in_hdf.keys())
        in_chunk = in_hdf[k[-1]].chunks

    xds = rioxarray.open_rasterio(in_path, 
                                  chunks={"x": in_chunk[0], "y": in_chunk[1]}, 
                                  masked=True)
    
    # Check CRS compatibility
    try:
        # Ensure the CRS match
        if region_shp.crs != xds.rio.crs:
            region_shp = region_shp.to_crs(xds.rio.crs)
    except CRSError as e:
        print(f"CRS error: {e}")
        return
    
    # Clip the raster
    clipped = xds.rio.clip(list(region_shp.geometry), region_shp.crs)
    clipped = clipped.where(clipped > 0, 0)
    
    # Multiply the urban_fraction with the pixel area
    clipped = clipped * ssp_urban_pix_km2
    
    meta = {'driver': 'GTiff', 
            'height': clipped.shape[1], 
            'width': clipped.shape[2], 
            'count': clipped.shape[0], 
            'dtype': str(clipped.values.dtype), 
            'crs': clipped.rio.crs, 
            'transform': clipped.rio.transform(),  
            'compress': 'lzw'}

    with rasterio.open(out_path, 'w', **meta) as dst:
        dst.write(clipped.values)




if __name__  == "__main__":
    
    region_shp = gpd.read_file('data/Vector_boundary/North_china_Plain_Province.shp')
    
    nc_files = glob('data/Urban_1km_1km/Raw/*.nc')
    out_path = [f"data/Urban_1km_1km/clipped/{os.path.basename(i).replace('nc','tiff')}" 
                for i in nc_files]
    
    for in_path, out_path in tqdm(zip(nc_files, out_path), total=len(nc_files)):
        clip_urban(in_path, out_path, region_shp)