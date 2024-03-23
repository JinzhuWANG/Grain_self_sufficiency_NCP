from glob import glob
from helper_func import get_img_df
import shutil

# get the path of GAEZ attainable yield
raw_tifs = r'C:\Users\Jinzhu\OneDrive - Deakin University\PHD Progress\Paper_3\Data\03_GAEZ\TIFs'
GAEZ_paths = glob(f'{raw_tifs}/*.tif')