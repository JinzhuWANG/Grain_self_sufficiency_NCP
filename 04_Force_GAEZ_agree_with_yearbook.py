import numpy as np
import pandas as pd
import rasterio

from data_transform.Force_GAEZ_with_yearbook import force_GAEZ_with_yearbook, get_GAEZ_df, get_each_and_total_value
from helper_func import ndarray_to_df
from helper_func.parameters import UNIQUE_VALUES




def GAEZ_to_yearbook(variable:str):
    
    # Read the yearbook data
    yearbook_base = 'yearbook_yield_targ_pc' if variable == 'Yield' else 'yearbook_area_targ_pc'
    out_name = 'yield' if variable == 'Yield' else 'area'


    # Get yearbook records
    yearbook_records_pc = np.load(f'data/results/{yearbook_base}.npy')
    yearbook_records_df = ndarray_to_df(yearbook_records_pc, 'pc')


    # Get GAEZ_df recording the tif_paths, and the mask corespoding the the calculation for the variable
    GAEZ_df, mask = get_GAEZ_df(var_type = variable)   
    # Get the data for each  and  total value of the variable based on water_supply
    GAEZ_arr_individual_cshw, GAEZ_arr_total_pc = get_each_and_total_value(GAEZ_df, variable, mask)
    # Get the GAEZ_arr that has been forced to agree with the yearbook data
    diff_pc, GAEZ_base_yr_pcshw = force_GAEZ_with_yearbook(GAEZ_arr_individual_cshw, GAEZ_arr_total_pc, yearbook_records_pc, mask)


    # Save to disk
    np.save(f'data/results/GAEZ_base_yr_{out_name}_pcshw.npy', GAEZ_base_yr_pcshw.astype(np.float16))
    
    
    # Save the GAEZ_base_yr_pcshw to a DataFrame
    GAEZ_base_yr_pcs = np.einsum('pcshw, phw->pcs', GAEZ_base_yr_pcshw, mask)              # (p, c, s)
    GAEZ_base_yr_pcs_df = ndarray_to_df(GAEZ_base_yr_pcs, 'pcs')
    
    # Merge the GAEZ_base_yr_pcs_df with the GAEZ_df
    GAEZ_yearbook_df = pd.merge(GAEZ_base_yr_pcs_df, 
                                yearbook_records_df,
                                on = ['crop', 'Province'], 
                                how = 'left',
                                suffixes = ('_GAEZ', '_yearbook'))


    return GAEZ_yearbook_df






# Sanity check
if __name__ == '__main__':

    results = {variable: GAEZ_to_yearbook(variable) for variable in ['Yield', 'Harvested area']}
        






