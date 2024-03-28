import numpy as np
import pandas as pd
import rasterio
import plotnine

from data_transform.Force_GAEZ_with_yearbook import force_GAEZ_with_yearbook, get_GAEZ_df, get_each_and_total_value
from helper_func import ndarray_to_df
from helper_func.parameters import BASE_YR, GAEZ_variables, GAEZ_water_supply, UNIQUE_VALUES


# Read the yearbook data
yearbook_yield = pd.read_csv('data/results/yield_yearbook.csv').sort_values(by=['Province', 'crop']).reset_index(drop=True)
yearbook_yield_targ = yearbook_yield.query(f'year == {BASE_YR}')
yearbook_yield_targ_long = yearbook_yield_targ['Yield (tonnes)'].values
yearbook_yield_targ_pc = yearbook_yield_targ_long.reshape(len(UNIQUE_VALUES['Province']),len(UNIQUE_VALUES['crop']))        # (p, c)

yearbook_area = pd.read_csv('data/results/yearbook_area.csv').sort_values(by=['Province', 'crop']).reset_index(drop=True)
yearbook_area_targ = yearbook_area.query(f'year == {BASE_YR}')
yearbook_area_targ_long = yearbook_area_targ['area_yearbook_kha'].values
yearbook_area_targ_pc = yearbook_area_targ_long.reshape(len(UNIQUE_VALUES['Province']),len(UNIQUE_VALUES['crop']))          # (p, c)


variable="Yield"

# Get yearbook records
yearbook_records_pc = yearbook_area_targ_pc if variable == 'Harvested area' else yearbook_yield_targ_pc



# Get GAEZ_df recording the tif_paths, and the mask corespoding the the calculation for the variable
GAEZ_df, mask = get_GAEZ_df(variable="Harvested area")   

# Get the data for each  and  total value of the variable based on water_supply
GAEZ_arr_individual_cshw, GAEZ_arr_total_chw = get_each_and_total_value(GAEZ_df, "Harvested area", mask)

# Get the GAEZ_arr that has been forced to agree with the yearbook data
diff_pc, GAEZ_base_yr_pcshw = force_GAEZ_with_yearbook(GAEZ_arr_individual_cshw, GAEZ_arr_total_chw, yearbook_records_pc, mask)

# Save to disk
np.save('data/results/GAEZ_base_yr_pcshw.npy', GAEZ_base_yr_pcshw.astype(np.float16))







in_path = 'data/GAEZ_v4/GAEZ_df.csv'
yearbook_targ_pc = yearbook_records_pc



# Read the GAEZ_df which records the metadata of the GAEZ data path
GAEZ_df = pd.read_csv(in_path)
GAEZ_df = GAEZ_df.query(f'GAEZ == "GAEZ_5" and variable == @variable')    
GAEZ_df = GAEZ_df.replace(GAEZ_water_supply['GAEZ_5']).infer_objects(copy=False)
GAEZ_df = GAEZ_df[GAEZ_variables['GAEZ_5'] + ['fpath']]
GAEZ_df = GAEZ_df.sort_values(by=['crop', 'water_supply']).reset_index(drop=True)



# Change the mask type based on variable type
if variable == 'Harvested area':
    mask_base = 'Province_mask'  
elif  variable == 'Yield':
    mask_base = 'Province_mask_mean'
else:
    raise ValueError('variable must be either "Harvested area" or "Province_mask"')  
with rasterio.open(f'data/GAEZ_v4/{mask_base}.tif') as src:
    mask = src.read()                                        # (p, h, w)









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
elif variable == 'Harvested area':
    GAEZ_arr_weighted_cshw = GAEZ_arr_cshw
else:
    raise ValueError('variable must be either "Harvested area" or "Yield"')

# Compute the total value of the variable
GAEZ_arr_total_chw = np.einsum('cshw->chw', GAEZ_arr_weighted_cshw)       









# Sum up all pixels for each province
GAEZ_arr_total_pc = np.einsum('chw,phw -> pc', GAEZ_arr_total_chw, mask)                # (p, c)

# Compare the defference (ratio) between yearbook and GAEZ
diff_pc = yearbook_targ_pc / GAEZ_arr_total_pc                                          # (p, c)
                                                                                
# Apply the diff to GAEZ_area_cshw
GAEZ_base_yr_pcshw = np.einsum('pc,cshw->pcshw', diff_pc, GAEZ_arr_individual_cshw)     # (p, c, s, h, w)




diff_pc_df = ndarray_to_df(diff_pc, 'pc')

GAEZ_arr_total_pc_df = ndarray_to_df(GAEZ_arr_total_pc, 'pc')
GAEZ_arr_total_chw_df = ndarray_to_df(GAEZ_arr_total_chw, 'chw')






if __name__ == '__main__':

    # Sanity check
    GAEZ_base_yr_pcs = np.einsum('pcshw->pcs', GAEZ_base_yr_pcshw)              # (p, c, s)
    GAEZ_base_yr_pcs_df = ndarray_to_df(GAEZ_base_yr_pcs, 'pcs')
    GAEZ_base_yr_pcs_df = GAEZ_base_yr_pcs_df.groupby(['Province','crop']).sum(numeric_only=True).reset_index()
    
    
    
    # Merge with yearbook_area_targ
    yearbook_GAEZ_area_df = pd.merge(GAEZ_base_yr_pcs_df, yearbook_area_targ, on=['Province','crop'], how='left')

    # Plot the yield for each province of both yearbook and GAEZ
    plotnine.options.figure_size = (16, 6)
    plotnine.options.dpi = 100
    g = (
        plotnine.ggplot() +
        plotnine.geom_point(yearbook_GAEZ_area_df,
                        plotnine.aes(x='Value', y='area_yearbook_kha', color='crop', shape='Province' )) +
        plotnine.geom_abline(plotnine.aes(intercept=0, slope=1), color="grey", linetype="dashed") +
        plotnine.theme_minimal() +
        plotnine.theme(axis_text_x=plotnine.element_text(rotation=45, hjust=1))
    )












