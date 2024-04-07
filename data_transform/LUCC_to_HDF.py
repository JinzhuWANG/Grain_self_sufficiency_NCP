
import os
import h5py
import numpy as np
import rasterio
from osgeo import gdal
from tqdm.auto import tqdm
from shapely.geometry import box

from helper_func.parameters import HDF_BLOCK_SIZE, RASTER_DICT





def read_tifs(tif_paths: str | list[str]) -> tuple[str, rasterio.DatasetReader]:
    fpath = ''
    # Read the tif file/s
    if isinstance(tif_paths, str):
        ds = rasterio.open(tif_paths)
    elif isinstance(tif_paths, list):
        # Build a vrt file from the list of tif files
        fname = os.path.basename(tif_paths[0])
        fname = os.path.splitext(fname)[0]
        fpath = f"data/LUCC/{fname}.vrt"
        
        vrt = gdal.BuildVRT(fpath, tif_paths)
        vrt.FlushCache()
        ds = rasterio.open(fpath)
    else:
        raise ValueError("tif_paths must be a string or a list of strings")
        
    return fpath, ds





def get_intersection_bounds(file_dict:dict) -> box:
    
    # Read the tif files
    raster_ds = [read_tifs(file)[1] for file in file_dict.values()]
    # Get the first file bounds
    intersection_bounds = box(*raster_ds[0].bounds)
    # If there is only one file, return the bounds of the first file
    if len(raster_ds) == 1:
        return intersection_bounds
    # Update the intersection bounds with the intersection of the current bounds and the bounds of each other file
    for ds in raster_ds[1:]:
        intersection_bounds = intersection_bounds.intersection(box(*ds.bounds))
            
    return intersection_bounds






def get_intersection_windows_trans(ds, intersection_box, block_size:int = HDF_BLOCK_SIZE):
    
    # Get the intersection bounds of the tif files
    intersection_bounds = intersection_box.bounds
    
    intersection_left = intersection_bounds[0]
    intersection_top = intersection_bounds[3]
    
    ds_left = ds.bounds[0]
    ds_top = ds.bounds[3]
    
    ds_col_width = ds.transform.a
    ds_row_height = ds.transform.e
    
    # Get the start row and column index of the intersection bounds
    intersection_row_start = int((intersection_top - ds_top ) // ds_row_height)
    intersection_col_start = int((intersection_left - ds_left) // ds_col_width)
    
    # Get the length and width of the intersection bounds in rows and columns
    intersection_box_rows =  int((intersection_bounds[1] - intersection_bounds[3]) // ds.transform.e)
    intersection_box_cols =  int((intersection_bounds[2] - intersection_bounds[0]) // ds.transform.a)
    
    # Get the end row and column index of the intersection bounds
    intersection_row_end = intersection_row_start + intersection_box_rows
    intersection_col_end = intersection_col_start + intersection_box_cols

    # Get the row and column ending-index of the intersection bounds
    block_row_idx = np.arange(block_size, intersection_box_rows, block_size)
    block_col_idx = np.arange(block_size, intersection_box_cols, block_size)
    
    
    # Split the row/column length by the block size
    row_blocks = np.array_split(np.arange(intersection_row_start, intersection_row_end), block_row_idx)
    col_blocks = np.array_split(np.arange(intersection_col_start, intersection_col_end), block_col_idx)
    
    # Create a list of windows
    windows_tif = [rasterio.windows.Window(col[0], row[0], len(col), len(row)) 
                    for row in row_blocks 
                    for col in col_blocks]
    
    # Create a list of windows for the hdf file
    windows_hdf = []
    for win in windows_tif:
        row_off = win.row_off - intersection_row_start
        col_off = win.col_off - intersection_col_start
        height = win.height
        width = win.width
        
        windows_hdf.append((row_off, row_off + height, col_off, col_off + width))
        
        
    # Create a transformation with the top left corner of the intersection bounds and the pixel size of the dataset
    intersection_transform = rasterio.transform.from_origin(intersection_bounds[0], 
                                                            intersection_bounds[3], 
                                                            ds.transform.a, 
                                                            -ds.transform.e)
    
    return windows_tif, windows_hdf, intersection_transform, (intersection_box_rows, intersection_box_cols)




def tif2hdf(tif: str | list[str], save_path: str, intersection_box, block_size:int = HDF_BLOCK_SIZE) -> None:
    
    # Get the fname of the tif file
    fname = os.path.basename(tif)
    # read the tif paths to a dataset
    fpath, ds = read_tifs(tif)
    # get dataset dtype
    ds_dtype = ds.dtypes[0]
    # get the intersection bounds of the tif files    
    windows_tif, windows_hdf, intersection_transform, ds_shape = get_intersection_windows_trans(ds, intersection_box)
    
    # Remove the shape of the intersection dataset
    print(f"Clip {fname} to shape: {ds_shape}")

    # create an hdf file for writing
    with h5py.File(save_path, mode='w') as hdf_file:
        
        # Create a dataset for the transformation list
        hdf_file.create_dataset('Transform', data=list(intersection_transform))
        # Create a dataset and save the NumPy array to it
        hdf_file.create_dataset('Array', 
                                shape=(ds.count,*ds_shape),
                                dtype=ds_dtype, 
                                fillvalue=0, 
                                compression="gzip", 
                                compression_opts=9,
                                chunks=(ds.count,block_size,block_size))


        # Loop through each window and read the data
        for idx, window_tif in tqdm(enumerate(windows_tif), total=len(windows_tif)): 
            
            window_hdf = windows_hdf[idx]        
            arr = ds.read(window=window_tif)
            # write the block arry to hdf
            hdf_file['Array'][:,
                        window_hdf[0] : window_hdf[1],
                        window_hdf[2] : window_hdf[3]] = arr

    # remove the fpath file
    os.remove(fpath) if os.path.exists(fpath) else None
                        
    return None



if __name__ == '__main__':

    for key, val in RASTER_DICT.items():
        tif2hdf(val, f"data/LUCC/{key}.hdf5", get_intersection_bounds(RASTER_DICT))
        
        
    # # Test datasets
    # small_ds = {'CLCD_v01_2019_small': 'data/LUCC/CLCD_v01_2019_small.tif',
    #             'Urban_1990_2019_small': 'data/LUCC/Urban_1990_2019_small.tif',
    #             'Transition_potential_small': 'data/LUCC/Transition_potential_small.tif'}
    
    # for key, val in small_ds.items():
    #     tif2hdf(val, f"data/LUCC/{key}.hdf5", get_intersection_bounds(small_ds))
                
                

    
        
    

   
        
