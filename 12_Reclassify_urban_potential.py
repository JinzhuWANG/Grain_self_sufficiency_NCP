import dask.array as da
import pandas as pd
import h5py
import rasterio
import rioxarray
import xarray as xr

from affine import Affine
from helper_func.parameters import HDF_BLOCK_SIZE

work_size = HDF_BLOCK_SIZE * 8

# Read data
potential_threshold = pd.read_csv('data/results/potential_threshold.csv')


urban_potential_arr = open_hdf_and_set_transform('data/LUCC/Transition_potential.hdf5', 'Array', 'Transform')

cropland_arr = open_hdf_and_set_transform('data/LUCC/CLCD_v01_2019.hdf5', 'Array', 'Transform')
cropland_arr = cropland_arr == 1 # Cropland is class 1

region_ds = h5py.File('data/LUCC/LUCC_Province_mask.hdf5', 'r')
region_arr = da.from_array(region_ds['Array'], chunks=(work_size, work_size))

lucc_area_ds = h5py.File('data/LUCC/LUCC_area_km2.hdf5', 'r')
lucc_area_area = da.from_array(lucc_area_ds['area'], chunks=(work_size, work_size))


 



def reproject_to_match_target(source_raster:str|xr.DataArray, 
                              target_raster:str|xr.DataArray):
    
    if isinstance(source_raster, str):
        source_raster = rioxarray.open_rasterio(source_raster)
    if isinstance(target_raster, str):
        target_raster = rioxarray.open_rasterio(target_raster)

    target_crs = target_raster.rio.crs
    target_transform = target_raster.rio.transform()
    target_width = target_raster.rio.width
    target_height = target_raster.rio.height

    return source_raster.rio.reproject(
        dst_crs=target_crs,
        shape=(target_height, target_width),
        transform=target_transform,
    )


def open_hdf_and_set_transform(file_path:str=None, 
                               dataset_name:str=None, 
                               transform_name:str='Transform',
                               crs:str="EPSG:4326"):
    
    # Open the HDF file and dataset
    with h5py.File(file_path, 'r') as data:
        data_arr = da.from_array(data[dataset_name], chunks=data[dataset_name].chunks)
        transform = Affine(*data[transform_name][:])

    # Handle potential extra dimension typically found in HDF datasets
    if data_arr.ndim == 3:
        data_arr = data_arr[0]

    # Create an xarray DataArray with coordinates
    xr_data = xr.DataArray( data_arr )

    # Set the CRS and transform using rioxarray
    xr_data.rio.write_crs(crs, inplace=True)
    xr_data.rio.write_transform(transform, inplace=True)

    return xr_data


GAEZ_xr = rioxarray.open_rasterio('data/GAEZ_v4/GAEZ_tifs/0a186d00-2931-4f31-b5c5-b6837dfeec51.tif', chunks=(4, 4))
Urban_xr = open_hdf_and_set_transform('data/LUCC/Urban_1990_2019.hdf5', 'Array', 'Transform')
urban_xr2 = rioxarray.open_rasterio(r'data\LUCC\North_China_Plain_1990_2019.tif', chunks=(work_size, work_size) )


GAEZ_rpj_xr = reproject_to_match_target(GAEZ_xr, urban_xr2)










