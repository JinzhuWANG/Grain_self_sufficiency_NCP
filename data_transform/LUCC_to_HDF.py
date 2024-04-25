
import os
import h5py
import numpy as np
import rasterio
from osgeo import gdal
from tqdm.auto import tqdm
from shapely.geometry import box

from helper_func.parameters import HDF_BLOCK_SIZE, RASTER_DICT




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





def get_intersection_box(file_dict:dict) -> box:
    """
    Calculates the intersection box and transformation for a set of raster files.

    Args:
        file_dict (dict): A dictionary containing file paths as values.

    Returns:
        tuple: A tuple containing the intersection box and transformation.
    """
    
    # Get the bounds of each file
    raster_ds = [read_tifs(file) for file in file_dict.values()]
    intersection_box = box(*raster_ds[0].bounds)
    
    # If there is only one file, return the bounds of the first file
    if len(raster_ds) == 1:
        return intersection_box, ds.transform
 
    # Update the intersection bounds with the intersection of the current bounds and the bounds of each other file
    for ds in raster_ds[1:]:
        intersection_box = intersection_box.intersection(box(*ds.bounds))
        
    # Create a transformation with the top left corner of the intersection bounds and the pixel size of the dataset
    intersection_transform = rasterio.transform.from_origin(intersection_box.bounds[0], 
                                                            intersection_box.bounds[3], 
                                                            ds.transform.a, 
                                                            -ds.transform.e)
            
    return intersection_box, intersection_transform






def get_clip_row_col(tif_path:str,  intersection_box: box):
    """
    Get the intersection windows, transformation, and dimensions for a given dataset and intersection box.
    
    Parameters:
    tif_path (str): The path to the TIFF file.
    intersection_box (box): The intersection box representing the bounds of the desired intersection.
    
    Returns:
    tuple: A tuple containing the start row index, number of rows, start column index, and number of columns
        for the intersection bounds.
    """
    
    ds = rasterio.open(tif_path)
    
    # Get the intersection bounds of the tif files
    intersection_bounds = intersection_box.bounds
    intersection_left = intersection_bounds[0]
    intersection_top = intersection_bounds[3]
    intersection_bottom = intersection_bounds[1]
    intersection_right = intersection_bounds[2]
    
    ds_top = ds.bounds[3]
    ds_left = ds.bounds[0]
    
    ds_pix_width = ds.transform.a
    ds_pix_height = ds.transform.e
    
    
    # Get the start row and column index of the intersection bounds
    intersection_row_start = (intersection_top - ds_top ) // ds_pix_height
    intersection_col_start = (intersection_left - ds_left) // ds_pix_width
    
    # Get the length and width of the intersection bounds in rows and columns
    intersection_rows =  round((intersection_top - intersection_bottom) / -ds.transform.e)
    intersection_cols =  round((intersection_right - intersection_left) / ds.transform.a)
    
    return intersection_row_start, intersection_rows, intersection_col_start, intersection_cols





def get_clip_windows(row_start:int, 
                     rows:int, 
                     col_start:int, 
                     cols:int, 
                     block_size:int = HDF_BLOCK_SIZE):
    """
    Generate clip windows for a given raster dataset.

    Args:
        row_start (int): The starting row index of the intersection bounds.
        rows (int): The number of rows in the intersection bounds.
        col_start (int): The starting column index of the intersection bounds.
        cols (int): The number of columns in the intersection bounds.
        block_size (int, optional): The size of each block for splitting the row/column length. 
                                   Defaults to BLOCK_SIZE.

    Returns:
        tuple: A tuple containing two lists of windows. The first list contains windows of type 
               `rasterio.windows.Window` representing the intersection bounds. The second list 
               contains windows of type tuple representing the clip bounds.

    """
    # Get the end row and column index of the intersection bounds
    row_end = row_start + rows
    col_end = col_start + cols
    
    # Get the row and column ending-index of the intersection bounds
    block_row_idx = np.arange(block_size, rows, block_size).astype(int)
    block_col_idx = np.arange(block_size, cols, block_size).astype(int)
    
    # Split the row/column length by the block size
    row_chunks = np.array_split(np.arange(row_start, row_end), block_row_idx)
    col_chunks = np.array_split(np.arange(col_start, col_end), block_col_idx)
    
    # Create a list of windows
    windows_from = [rasterio.windows.Window(col[0], row[0], len(col), len(row)) 
                    for row in row_chunks 
                    for col in col_chunks]
    
    # Create a list of windows for the cliped file
    windows_to = [rasterio.windows.Window(col[0] - col_start, row[0] - row_start, len(col), len(row)) 
                    for row in row_chunks 
                    for col in col_chunks]

    return windows_from, windows_to





def tif2hdf(fname:str,
            tif_path: str, 
            intersection_transform: rasterio.transform.Affine,
            intersection_rows: int,
            intersection_cols: int,
            windows_from:list[rasterio.windows.Window], 
            windows_to:list[rasterio.windows.Window],
            block_size:int=HDF_BLOCK_SIZE) -> None:
    
    # Read the tif paths to a dataset
    ds = rasterio.open(tif_path)
    ds_dtype = ds.dtypes[0]
    save_path = f'{os.path.split(tif_path)[0]}/{fname}_cliped.hdf'

    
    # Report the shape of the intersection dataset
    print(f"Clip {fname} to shape: {(int(intersection_rows), int(intersection_cols))}")

    # create an hdf file for writing
    with h5py.File(save_path, mode='w') as hdf_file:
    
        # Create a dataset for the transformation list
        hdf_file.create_dataset('Transform', data=list(intersection_transform))
        # Create a dataset and save the NumPy array to it
        hdf_file.create_dataset('Array', 
                                shape=(ds.count,intersection_rows,intersection_cols),
                                dtype=ds_dtype, 
                                fillvalue=0, 
                                compression="gzip", 
                                compression_opts=9,
                                chunks=(ds.count, block_size, block_size))

        # Create the cliped hdf file
        for win_from, win_to in tqdm(zip(windows_from, windows_to), total=len(windows_from)):      
            arr = ds.read(window=win_from)
            # write the block arry to hdf
            hdf_file['Array'][:,
                        slice(int(win_to.row_off), int(win_to.row_off) + win_to.height),
                        slice(int(win_to.col_off), int(win_to.col_off) + win_to.width)] = arr
                        
    return None



if __name__ == '__main__':

        
    # Test datasets
    tif_dict = {'CLCD_v01_2019_small': 'data/LUCC/CLCD_v01_2019_small.tif',
                'Urban_1990_2019_small': 'data/LUCC/Urban_1990_2019_small.tif',
                'Transition_potential_small': 'data/LUCC/Transition_potential_small.tif'}
    
    # Real datasets
    tif_dict = RASTER_DICT
    

    # Get the intersection box and transformation
    intersection_box, intersection_transform = get_intersection_box(tif_dict)
    
    for fname, tif in tif_dict.items():
    
        intersection_row_start, intersection_rows, intersection_col_start, intersection_cols = get_clip_row_col(tif, intersection_box)
        windows_from, windows_to = get_clip_windows(intersection_row_start, intersection_rows, intersection_col_start, intersection_cols)
        tif2hdf(fname,
                tif, 
                intersection_transform, 
                intersection_rows, 
                intersection_cols,
                windows_from,
                windows_to
                )
        

                

    
        
    

   
        
