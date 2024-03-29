import numpy as np
import pandas as pd
import plotnine

from data_transform.Force_GAEZ_with_yearbook import force_GAEZ_with_yearbook, get_GAEZ_df, get_each_and_total_value
from helper_func import ndarray_to_df
from helper_func.parameters import BASE_YR, GAEZ_variables, GAEZ_water_supply, UNIQUE_VALUES




def GAEZ_to_yearbook(variable:str):
    
    # Read the yearbook data
    yearbook_base = 'yearbook_yield_targ_pc' if variable == 'Yield' else 'yearbook_area_targ_pc'
    out_name = 'yield' if variable == 'Yield' else 'area'

    # Get yearbook records
    yearbook_records_pc = np.load(f'data/results/{yearbook_base}.npy')
    yearbook_records_df = ndarray_to_df(yearbook_records_pc, 'pc')


    # Get GAEZ_df recording the tif_paths, and the mask corespoding the the calculation for the variable
    GAEZ_df, mask = get_GAEZ_df(variable="Harvested area")   
    # Get the data for each  and  total value of the variable based on water_supply
    GAEZ_arr_individual_cshw, GAEZ_arr_total_chw = get_each_and_total_value(GAEZ_df, "Harvested area", mask)
    # Get the GAEZ_arr that has been forced to agree with the yearbook data
    GAEZ_base_yr_pcshw = force_GAEZ_with_yearbook(GAEZ_arr_individual_cshw, GAEZ_arr_total_chw, yearbook_records_pc, mask)

    # Save to disk
    np.save(f'data/results/GAEZ_base_yr_{out_name}_pcshw.npy', GAEZ_base_yr_pcshw.astype(np.float16))
    
    
    
    # Save the GAEZ_base_yr_pcshw to a DataFrame
    GAEZ_base_yr_pc = np.einsum('pcshw, phw->pc', GAEZ_base_yr_pcshw, mask)              # (p, c)
    GAEZ_base_yr_pc_df = ndarray_to_df(GAEZ_base_yr_pc, 'pc')
    
    
    # Merge with yearbook records
    yearbook_GAEZ_df = pd.merge(GAEZ_base_yr_pc_df, 
                                yearbook_records_df, 
                                on=['Province','crop'], 
                                how='left', 
                                suffixes=('_GAEZ', '_Yearbook'))
    
    
    return yearbook_GAEZ_df





# Sanity check
if __name__ == '__main__':

    results = {variable: GAEZ_to_yearbook(variable) for variable in ['Yield', 'Harvested area']}
        











