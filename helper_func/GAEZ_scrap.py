import time
import requests
from tqdm.auto import tqdm
import pandas as pd


headers = {'user-agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) " \
                            "AppleWebKit/537.36 (KHTML, like Gecko) " \
                            "Chrome/101.0.4951.67 Safari/537.36"}



# get the total number of input images, 
# these numbers comes from the GAEZ website <https://gaez.fao.org/pages/data-viewer>
# by inspecting the HTML elements
total_count = {
    'GAEZ_1': 134,
    'GAEZ_2': 78245,
    'GAEZ_3': 111672,
    'GAEZ_4': 122672,
    'GAEZ_5': 498,
    'GAEZ_6': 288
    }

# Get the GAEZ query load, these urls comes from the GAEZ website <https://gaez.fao.org/pages/data-viewer>
urls = {
'GAEZ_1' : 'https://gaez-services.fao.org/server/rest/services/LR/ImageServer/query?f=json&where=((1%3D1))&spatialRel=esriSpatialRelIntersects&geometry=%7B%22xmin%22%3A-15438160.25137524%2C%22ymin%22%3A-6208804.384303834%2C%22xmax%22%3A-253485.96035930235%2C%22ymax%22%3A11363151.174114093%2C%22spatialReference%22%3A%7B%22wkid%22%3A102100%7D%7D&geometryType=esriGeometryEnvelope&inSR=102100&outFields=OBJECTID%2CName%2Cfile_id%2Csub_theme_name%2Cvariable%2Cfile_description%2Cyear%2Cmodel%2Crcp%2Cunits%2Crenderer%2Cdownload_url%2CThumbnail%2CMinPS%2CMaxPS%2CLowPS%2CHighPS%2CTag%2CGroupName%2CProductName%2CCenterX%2CCenterY%2CZOrder%2CShape_Length%2CShape_Area%2Cfilepath&orderByFields=OBJECTID%20ASC&outSR=102100',
'GAEZ_2' : 'https://gaez-services.fao.org/server/rest/services/res01/ImageServer/query?f=json&where=((1%3D1))&returnGeometry=false&spatialRel=esriSpatialRelIntersects&geometry=%7B%22xmin%22%3A-16071667.993656825%2C%22ymin%22%3A-15823972.601728432%2C%22xmax%22%3A14297680.588375252%2C%22ymax%22%3A19319938.515107654%2C%22spatialReference%22%3A%7B%22wkid%22%3A102100%7D%7D&geometryType=esriGeometryEnvelope&inSR=102100&outFields=*&orderByFields=objectid%20ASC&outSR=102100&resultOffset={}&resultRecordCount={}',
'GAEZ_3' : 'https://gaez-services.fao.org/server/rest/services/res02/ImageServer/query?f=json&where=((1%3D1))&returnGeometry=false&spatialRel=esriSpatialRelIntersects&outFields=*&orderByFields=objectid%20ASC&resultOffset={}&resultRecordCount={}',
'GAEZ_4' : 'https://gaez-services.fao.org/server/rest/services/res05/ImageServer/query?f=json&where=((1%3D1))&returnGeometry=false&spatialRel=esriSpatialRelIntersects&geometry=%7B%22xmin%22%3A-16772704.574167874%2C%22ymin%22%3A-15824058.820688367%2C%22xmax%22%3A13596644.007864203%2C%22ymax%22%3A19319852.29614772%2C%22spatialReference%22%3A%7B%22wkid%22%3A102100%7D%7D&geometryType=esriGeometryEnvelope&inSR=102100&outFields=*&orderByFields=objectid%20ASC&outSR=102100&resultOffset={}&resultRecordCount={}',
'GAEZ_5' : 'https://gaez-services.fao.org/server/rest/services/res06/ImageServer/query?f=json&where=((1%3D1))&spatialRel=esriSpatialRelIntersects&geometry=%7B%22xmin%22%3A-19548086.635538064%2C%22ymin%22%3A-16896309.58729416%2C%22xmax%22%3A10821261.946494006%2C%22ymax%22%3A18247601.529541925%2C%22spatialReference%22%3A%7B%22wkid%22%3A102100%7D%7D&geometryType=esriGeometryEnvelope&inSR=102100&outFields=OBJECTID%2CName%2Cfile_id%2Csub_theme_name%2Cvariable%2Cfile_description%2Cyear%2Ccrop%2Cwater_supply%2Cunits%2Crenderer%2Cdownload_url&orderByFields=OBJECTID%20ASC&outSR=102100',
'GAEZ_6' : 'https://gaez-services.fao.org/server/rest/services/res07/ImageServer/query?f=json&where=((1%3D1))&spatialRel=esriSpatialRelIntersects&geometry=%7B%22xmin%22%3A-16199891.349127054%2C%22ymin%22%3A-15790365.759673677%2C%22xmax%22%3A14169457.232905015%2C%22ymax%22%3A19353545.35716241%2C%22spatialReference%22%3A%7B%22wkid%22%3A102100%7D%7D&geometryType=esriGeometryEnvelope&inSR=102100&outFields=OBJECTID%2CName%2Cfilepath%2Csub_theme_name%2Cvariable%2Cfile_description%2Cyear%2Ccrop%2Cwater_supply%2Cunits%2Crenderer%2Cdownload_url%2Cfile_id&orderByFields=OBJECTID%20ASC&outSR=102100'
}



def get_with_retry(get_url, headers, max_retries=5):
    for i in range(max_retries):
        try:
            response = requests.get(get_url, headers=headers)
            response.raise_for_status()  # Raises a HTTPError if the status is 4xx, 5xx
            return response
        except requests.exceptions.RequestException as e:
            print(f"Request failed with {e}. Retrying...")
            time.sleep(2)  # Wait for 2 seconds before retrying
    print(f"Failed to fetch the URL after {max_retries} attempts.")
    return None





def get_GAEZ_download_csv(url,total_len):
    
    data_retrive = []
    
    # if the total records is less than 1000, then just get them at once
    if total_len < 1000:
        re = requests.get(url,headers=headers).json()
        re_dicts = [i['attributes'] for i in re['features']]
        data_retrive.extend(re_dicts)
        return data_retrive
    
    # if the total records is largers than 1000, then splits it to 1000-sized chunk
    block = int(total_len/1000)
    remainder = total_len%1000
    for i in tqdm(range(block+1)):

        # change get_lenth judged by if this is the last block
        if i+1 == block:
            length = remainder
        else:
            length = 1000
            
        # get the start point   
        start = i*1000
        get_url = url.format(start,length)
        
        # get data
        re = get_with_retry(get_url,headers=headers).json()
        re_dicts = [i['attributes'] for i in re['features']]
        data_retrive.extend(re_dicts)
        
    return data_retrive




def get_all_GAEZ_urls():
    
    all_data = []
    for theme in range(1,7):
        theme_n = f'GAEZ_{theme}'
        url = urls[theme_n]
        total_len = total_count[theme_n]
        
        result = get_GAEZ_download_csv(url,total_len)
        
        df = pd.DataFrame(result)
        df.to_csv(f'data/GAEZ_v4/GAEZ_raw_urls/{theme_n}.csv', index=False)
        print(f"{theme_n} finished!")
        
    return all_data


if __name__ == '__main__':
    get_all_GAEZ_urls()

