import affine
import os
import rasterio
from osgeo import gdal
from shapely.geometry import box

from helper_func.parameters import HDF_BLOCK_SIZE

from rasterio.enums import Resampling
from rasterio import shutil as rio_shutil
from rasterio.vrt import WarpedVRT



# Function to merge multiple tif tiles into a virtual raster file
def tif2vir(tif_paths):
    # Build a vrt file from the list of tif files
    fname = os.path.basename(tif_paths[0])
    fname = os.path.splitext(fname)[0]
    fpath = f"data/LUCC/{fname}.vrt"
    
    vrt = gdal.BuildVRT(fpath, tif_paths)
    vrt.FlushCache()
    return rasterio.open(fpath)



def read_tifs(tif_paths: str | list[str]) -> tuple[str, rasterio.DatasetReader]:
    # Read the tif file/s
    if isinstance(tif_paths, str):
        ds = rasterio.open(tif_paths)
    elif isinstance(tif_paths, list):
        ds = tif2vir(tif_paths)
    else:
        raise ValueError("tif_paths must be a string or a list of strings")
    return ds





def get_warp_options(files: list[str|list]) -> box:
    """
    Calculate the warp options for a list of raster files.

    Args:
        files (list[str|list]): A list of file paths or lists of file paths.

    Returns:
        dict: A dictionary containing the warp options.

    """
    
    # Get the bounds of each file
    raster_ds = [read_tifs(file) for file in files]
    ds_first = raster_ds[0]
    intersection_box = box(*raster_ds[0].bounds)
    
    # If there is only one file, return the bounds of the first file
    if len(raster_ds) == 1:
        return intersection_box, ds.transform
 
    # Update the intersection bounds with the intersection of the current bounds and the bounds of each other file
    for ds in raster_ds[1:]:
        intersection_box = intersection_box.intersection(box(*ds.bounds))
        
    left, bottom, right, top = intersection_box.bounds
    xres = (right - left) / ds_first.width
    yres = (top - bottom) / ds_first.height
    dst_transform = affine.Affine(xres, 0.0, left,
                                0.0, -yres, top)
            
    return {
        'resampling': Resampling.nearest,
        'crs': ds_first.crs,
        'transform': dst_transform,
        'height': ds_first.height,
        'width': ds_first.width
    }



if __name__ == '__main__':

    files_dict = {
        'Norm_Urban_1990_2019' : 'data/LUCC/North_China_Plain_1990_2019.tif',
        'Norm_CLCD_v01_2019' : 'data/LUCC/CLCD_v01_2019.tif',
        'Norm_Transition_potential': ['data/LUCC/Transition_Potential-0000000000-0000000000.tif',
                                     'data/LUCC/Transition_Potential-0000046592-0000000000.tif'],
    }

    warp_option = get_warp_options(files_dict.values())
    copy_option = {'compress': 'lzw',
        'tiled': True,
        'blockxsize': HDF_BLOCK_SIZE,
        'blockysize': HDF_BLOCK_SIZE
    }

    for k,v in files_dict.items():
        ds = read_tifs(v)
        dtype = ds.dtypes[0]
        with WarpedVRT(ds, **warp_option) as vrt:
            rio_shutil.copy(vrt, f"data/LUCC/{k}.tif", driver='GTiff', dtype=dtype, **copy_option)
        


        

                

    
        
    

   
        
