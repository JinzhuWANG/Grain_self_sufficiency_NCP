import numpy as np

from data_transform.Force_GAEZ_with_yearbook import get_GAEZ_df, get_each_and_total_value
from helper_func import ndarray_to_df




# Read the GAEZ_area_df
GAEZ_df, mask = get_GAEZ_df(var_type = 'Harvested area')                                    # (p, h, w)
# Get the area for each water_supply
GAEZ_arr_individual_cshw, _ = get_each_and_total_value(GAEZ_df, 'Harvested area', mask)     # (c, s, h, w)
# Get the area for each province, crop, water_supply
GAEZ_area_yb_base_yr_pcs =  np.einsum('cshw,phw->pcs', GAEZ_arr_individual_cshw, mask)      # (p, c, s)



# Compute the total area 
GAEZ_area_yb_base_yr_pc = np.einsum('pcs->pc', GAEZ_area_yb_base_yr_pcs)                    # (p, c)
# Divide each crop-water_supply by its total area
GAEZ_crop_ratio_pcs = GAEZ_area_yb_base_yr_pcs / GAEZ_area_yb_base_yr_pc[:, :, None]        # (p, c, s)
GAEZ_crop_ratio_pcs_df = ndarray_to_df(GAEZ_crop_ratio_pcs, 'pcs')


# Save to disk
np.save('data/results/GAEZ_crop_ratio_pcs.npy', GAEZ_crop_ratio_pcs.astype(np.float16))




if __name__ == '__main__':
        
    # Sanity check
    GAEZ_crop_ratio_pcs_df = ndarray_to_df(GAEZ_crop_ratio_pcs, 'pcs')