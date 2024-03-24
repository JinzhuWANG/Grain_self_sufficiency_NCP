import sys
import uuid
import concurrent

import pandas as pd
from glob import glob
from tqdm.auto import tqdm

sys.path.append('./')

from helper_func.GAEZ_scrap import get_with_retry, headers
from helper_func.parameters import (crops, 
                                    GAEZ_variables, 
                                    GAEZ_columns, 
                                    GAEZ_input,
                                    GAEZ_years,
                                    GAEZ_water_supply,
                                    PARALLEY_THREADS)



def download_url(url, fpath):
    # Send a GET request to the URL
    response = get_with_retry(url, headers=headers)
    
    if response != None:
        # Open the file in write mode
        with open(fpath, 'wb') as f:
            f.write(response.content)
    else:
        print(f"Failed to download data from {url}")



# Download the GAEZ data
def download_GAEZ_data(GAEZ_df):
    with concurrent.futures.ThreadPoolExecutor(PARALLEY_THREADS) as executor:
        futures = []
        fpaths = []
        for idx, row in GAEZ_df.iterrows():
            unique_id = uuid.uuid4()
            fname = f"data/GAEZ_v4/GAEZ_tifs/{unique_id}.tif"
            url = row['download_url']
            
            futures.append(executor.submit(download_url, url, fname))
            fpaths.append(fname)

        # Execute the futures as they are completed, report the process
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            try:
                future.result()
            except Exception as exc:
                print(f'Generated an exception: {exc}')
                
        # Append the fpaths to the GAEZ_df
        GAEZ_df['fpath'] = fpaths
        
        # Save the GAEZ_df
        GAEZ_df.to_csv('data/GAEZ_v4/GAEZ_df.csv', index = False)
        
        print("GAEZ data downloaded successfully!")
    




if __name__ == '__main__':
    
    # Read the GAEZ_url file
    GAEZ_df = []
    for GAEZ in GAEZ_input:
        # Read the csv file
        df = pd.read_csv(f'data/GAEZ_v4/GAEZ_raw_urls/{GAEZ}.csv').rename(columns = {'Name':'name'})
        # Select the columns
        df = df[GAEZ_columns[GAEZ]]
        # Filter the rows
        df = df.query(f'crop in {crops} and year in {GAEZ_years[GAEZ]} and variable in {GAEZ_variables[GAEZ]}')
        # Rename the water_supply column
        df['water_supply'] = df['water_supply'].replace(GAEZ_water_supply[GAEZ])
        # Add the GAEZ category
        df.insert(0, 'GAEZ', GAEZ)
        
        GAEZ_df.append(df)
    
    # Concatenate the GAEZ_df    
    GAEZ_df = pd.concat(GAEZ_df).reset_index(drop = True)
    
    # Download the GAEZ data
    download_GAEZ_data(GAEZ_df)

    
