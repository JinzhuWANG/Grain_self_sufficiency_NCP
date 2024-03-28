import numpy as np

from helper_func import ndarray_to_df


# Read the GAEZ_area_yb_pcshw_base_yr
GAEZ_area_yb_base_yr_pcshw = np.load('data/results/GAEZ_area_yb_base_yr_pcshw.npy')         # (p, c, s, h, w)

# Compute the total area 
GAEZ_area_yb_base_yr_pcs = np.einsum('pcshw->pcs', GAEZ_area_yb_base_yr_pcshw)              # (p, c, s)
GAEZ_area_yb_base_yr_pc = np.einsum('pcs->pc', GAEZ_area_yb_base_yr_pcs)                    # (p, c)

# Divide each crop-water_supply by its total area
GAEZ_crop_ratio_pcs = GAEZ_area_yb_base_yr_pcs / GAEZ_area_yb_base_yr_pc[:, :, None]        # (p, c, s)

# Save to disk
np.save('data/results/GAEZ_crop_ratio_pcs.npy', GAEZ_crop_ratio_pcs.astype(np.float16))


if __name__ == '__main__':
        
    # Sanity check
    GAEZ_crop_ratio_pcs_df = ndarray_to_df(GAEZ_crop_ratio_pcs, 'pcs')