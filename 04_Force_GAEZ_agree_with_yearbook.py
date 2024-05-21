import numpy as np
import pandas as pd
import rioxarray as rxr
import xarray as xr

from helper_func.calculate_GAEZ_stats import get_GAEZ_df, get_GAEZ_stats
from helper_func.get_yearbook_records import get_yearbook_yield


# Get data
yearbook_yield = get_yearbook_yield().query('year == 2020').reset_index(drop=True)
GAEZ_yield_df = get_GAEZ_df(var_type = 'Yield')

GAEZ_area_stats = get_GAEZ_stats('Harvested area')


xr_arrs = []
for _, row in GAEZ_yield_df.iterrows():
    path = row['fpath']
    crop = row['crop']
    water_supply = row['water_supply']

    xr_arr = rxr.open_rasterio(path).squeeze()
    xr_arr = xr_arr.expand_dims({'crop': [crop], 'water_supply': [water_supply]})
    xr_arrs.append(xr_arr)

GAEZ_yield_xr = xr.combine_by_coords(xr_arrs)

# Get the GAEZ_yield stats
mask_mean = rxr.open_rasterio('data/GAEZ_v4/province_mask_mean.tif').squeeze('band')
mask_sum = rxr.open_rasterio('data/GAEZ_v4/province_mask.tif').squeeze('band')



GAEZ_yield_stats.name = 'Yield'
GAEZ_yield_stats = GAEZ_yield_stats.to_dataframe().reset_index()











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
        






