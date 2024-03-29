import numpy as np
import pandas as pd
import rasterio

from helper_func.parameters import UNIQUE_VALUES, GAEZ_variables, GAEZ_water_supply


def get_GAEZ_df(in_path:str = 'data/GAEZ_v4/GAEZ_df.csv', var_type:str = 'Harvested area'):

    # Read the GAEZ_df which records the metadata of the GAEZ data path
    GAEZ_df = pd.read_csv(in_path)
    GAEZ_df = GAEZ_df.query(f'GAEZ == "GAEZ_5" and variable == @var_type')    
    GAEZ_df = GAEZ_df.replace(GAEZ_water_supply['GAEZ_5'])
    GAEZ_df = GAEZ_df[GAEZ_variables['GAEZ_5'] + ['fpath']]
    GAEZ_df = GAEZ_df.sort_values(by=['crop', 'water_supply']).reset_index(drop=True)


    # Change the mask type based on variable type
    if var_type == 'Harvested area':
        mask_base = 'Province_mask'  
    elif  var_type == 'Yield':
        mask_base = 'Province_mask_mean'
    else:
        raise ValueError('variable must be either "Harvested area" or "Province_mask"')
        
    with rasterio.open(f'data/GAEZ_v4/{mask_base}.tif') as src:
        mask = src.read()                                        # (p, h, w)
        
    return GAEZ_df, mask



def get_each_and_total_value(GAEZ_df:pd.DataFrame, variable:str, mask:np.ndarray):
    
    # Read the data of <water_supply != Total>
    GAEZ_paths = GAEZ_df.query('water_supply != "Total"')['fpath'].tolist()

    GAEZ_arr_rhw = np.stack([rasterio.open(fpath).read(1) for fpath in GAEZ_paths], 0).flatten()                   # (c * s * h * w)
    GAEZ_arr_cshw = GAEZ_arr_rhw.reshape(len(UNIQUE_VALUES['crop']),
                                            len(UNIQUE_VALUES['water_supply']),
                                            *mask.shape[1:])                                                       # (c, s, h, w)

    # The arr need to be weighted by the area_propotion, if the arr is Yield
    if variable == 'Yield':
        GAEZ_crop_ratio_pcs = np.load('data/results/GAEZ_crop_ratio_pcs.npy')                                      # (p, c, s)
        GAEZ_arr_weighted_cshw = np.einsum('cshw,pcs->cshw', GAEZ_arr_cshw, GAEZ_crop_ratio_pcs)                   # (p, c, s, h, w)
        GAEZ_arr_weighted_cshw = GAEZ_arr_weighted_cshw / len(UNIQUE_VALUES['Province'])                           # (p, c, s, h, w)
    elif variable == 'Harvested area':
        GAEZ_arr_weighted_cshw = GAEZ_arr_cshw
    else:
        raise ValueError('variable must be either "Harvested area" or "Yield"')


    # Compute the total value of the variable
    GAEZ_arr_total_pc = np.einsum('cshw,phw->pc', GAEZ_arr_weighted_cshw, mask)                              # (c, h, w)
    
    return GAEZ_arr_cshw, GAEZ_arr_total_pc




def force_GAEZ_with_yearbook(GAEZ_arr_individual_cshw:np.ndarray, 
                             GAEZ_arr_total_pc:np.ndarray, 
                             yearbook_targ_pc:np.ndarray, 
                             mask:np.ndarray):


    # Compare the defference (ratio) between yearbook and GAEZ
    diff_pc = yearbook_targ_pc / GAEZ_arr_total_pc                                          # (p, c)                                                                              
    # Apply the diff to GAEZ_area_cshw
    GAEZ_base_yr_pcshw = np.einsum('pc,cshw->pcshw', diff_pc, GAEZ_arr_individual_cshw)     # (p, c, s, h, w)
    
    
    return  diff_pc, GAEZ_base_yr_pcshw