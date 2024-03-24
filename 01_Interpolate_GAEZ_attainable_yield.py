import os
from glob import glob
import shutil
import pandas as pd

from helper_func import get_img_df


# get the path of GAEZ attainable yield
raw_tifs = r'C:\Users\Jinzhu\OneDrive - Deakin University\PHD Progress\Paper_3\Data\03_GAEZ\TIFs'
GAEZ_paths = glob(f'{raw_tifs}/*.tif')
GAEZ_names = [os.path.basename(i).split('.')[0] for i in GAEZ_paths]



# get the historical imgs
img_Maize = get_img_df(img_path = GAEZ_paths,
          theme = 'GAEZ_4',
          attain_type = 'current',
          crop = ['Maize'],
          time = ['1990','2000','2010',],
          climate_model = 'CRUTS32',
          rcp = ['Historical'],
          co2 = 'With_CO2',
          input_level = 'High',
          water = ['Rainfed','Sprinkler'],
          exclusion = 'All'
          )
