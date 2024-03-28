import os
import sys
import pandas
import rasterio
import geopandas as gpd
import concurrent
import concurrent.futures

from rasterio.mask import mask
from tqdm.auto import tqdm

sys.path.append('./')

from helper_func.calculate_WGS_pix_area_km2 import calculate_area
from helper_func.parameters import PARALLEY_THREADS


    
# Clip the GAEZ data to the research region
def clip_GAEZ(row: pandas.Series, mask_shp: gpd.GeoDataFrame):
    with rasterio.open(row['fpath'], 'r') as src:
        out_image, out_transform = mask(src, mask_shp.geometry, crop=True)
        out_meta = src.meta.copy()
        out_meta.update({"driver": "GTiff",
                         "height": out_image.shape[1],
                         "width": out_image.shape[2],
                         "transform": out_transform,
                         "compress": "lzw"})
        
        # Get the fpath_clip
        fparent, fbase = os.path.split(row['fpath'])
        fpath_clip = f"{fparent}/clipped_{fbase}"
        
        # Save the clipped GAEZ data
        with rasterio.open(fpath_clip, "w", **out_meta) as dest:
            dest.write(out_image)
            
    # Remove the original GAEZ data, rename the clipped GAEZ data to the original name
    os.remove(row['fpath'])
    os.rename(fpath_clip, row['fpath'])
        
        
        
if __name__ == "__main__":
    # Read the GAEZ_df which records the metadata of the GAEZ data
    GAEZ_df = pandas.read_csv('data/GAEZ_v4/GAEZ_df.csv')

    # Read the shp of the research region
    research_region_shp = gpd.read_file('data/Vector_boundary/North_china_Plain_Province.shp')

    # Check if the crs of the GAEZ and shp are the same
    with rasterio.open(GAEZ_df.iloc[0]['fpath']) as src:
        GAEZ_crs = src.crs
        if GAEZ_crs != research_region_shp.crs:
            raise ValueError("The crs of the GAEZ and shp are not the same!")
        
    # Clip the GAEZ data to the research region
    with concurrent.futures.ThreadPoolExecutor(max_workers=PARALLEY_THREADS) as executor:
        tasks = []
        for idx, row in GAEZ_df.iterrows():
            tasks.append(executor.submit(clip_GAEZ, row, research_region_shp))
            
        for future in tqdm(concurrent.futures.as_completed(tasks), total=len(tasks)):
            try:
                future.result()
            except Exception as exc:
                print(f'Generated an exception: {exc}')
        
    
        