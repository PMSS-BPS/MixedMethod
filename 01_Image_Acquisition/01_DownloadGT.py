import asf_search as asf
import geopandas as gpd
from shapely.geometry import box
from datetime import date
import time,sys
import json
from tqdm import tqdm
import warnings
import numpy as np
import os
warnings.filterwarnings("ignore")
def read_json(path):
    with open(path, 'r') as file:
        data = json.load(file)
    return data

def get_file_size(file_path):
    """Get the size of a file in bytes."""
    return os.path.getsize(file_path)
def write_json(idprov, dict_list):
    path_file = f'/data/ksa/01_Image_Acquisition/05_Json_Check_Download/{idprov}_groundtruth_download_ASF.json'
    with open(path_file, 'w') as json_file:
        json.dump(dict_list, json_file, ensure_ascii=False, indent=4)
def remove_file(file_path):
    """Remove a file."""
    try:
        os.remove(file_path)
    except FileNotFoundError:
        print(f"File {file_path} not found.")
    except PermissionError:
        print(f"Permission denied to remove file {file_path}.")
    except Exception as e:
        print(f"Error occurred while removing file {file_path}: {e}")
def remove_corrupt(list_file):
    for i in list_file:
        remove_file(i)

def docheck(path_file,kdkab):
    data = read_json(path_file)
    dict_result = {}
    undownloaded_files = []
    downloaded_files = []
    corrupt_files = []
    for item in data['features']:
        filename = item['properties']['fileName'][:-4]
        # Check if file exists
        path_file = f'/data/ksa/01_Image_Acquisition/01_Raw_Image/{filename}.zip'
        if not os.path.exists(path_file):
            undownloaded_files.append(filename)
        else:
            size_file = get_file_size(path_file)
            if size_file != item['properties']['bytes']:
                corrupt_files.append(path_file)
            else:
                downloaded_files.append(filename)
    dict_result['undownloaded'] = undownloaded_files
    dict_result['corrupted'] = corrupt_files
    dict_result['downloaded'] = downloaded_files
    print(f'Downloaded data for {kdkab}: ',len(dict_result['downloaded']),
          '; Undownloaded/Corrupted',len(dict_result['corrupted'])+len(dict_result['undownloaded']))
    write_json(kdkab, dict_result)
    if len(undownloaded_files) == 0 and len(corrupt_files) == 0:
        print("Data Image sudah selesai didownload. Silakan lanjut ke proses selanjutnya")
    elif len(corrupt_files) > 0:
        remove_corrupt(corrupt_files)
        print("Masih terdapat file yang belum sempurna didownload. Harap running ulang kodingan download file")
    else:
        print("Masih terdapat data image yang belum didownload")
    return len(undownloaded_files)+len(corrupt_files)
def dodownload(kdkab,bound):
    print('Begin downloading at ',time.time())
    start_time=time.time()
    print('Downloading for' ,kdkab)
    wkt_aoi = bound.wkt
    asf.CMR_TIMEOUT = 60
    results = asf.search(
        platform= asf.PLATFORM.SENTINEL1A,
        processingLevel=[asf.PRODUCT_TYPE.GRD_HD],
        start = date(2024, 9, 1),
        end = date(2024,9 , 30),
        intersectsWith = wkt_aoi,
        )
    print(f'Total Images Found: {len(results)}')
    metadata = results.geojson()
    json_object = json.dumps(metadata)
    print('Writing the metadata.......')
    with open(f'/data/ksa/01_Image_Acquisition/04_Json_Raw_Download/{kdkab}_groundtruth_metadata_ASF.json', 'w') as f:
        f.write(json_object)
    with open('config.txt', 'r') as file:
        token = file.read().rstrip()
        session=asf.ASFSession().auth_with_token(token)
        dt=int(np.ceil(len(results)/1))
        print(dt)
        try:
            results.download(
                path = '/data/ksa/01_Image_Acquisition/01_Raw_Image',
                session = session,
                processes = dt)
        except Exception as e:
            print(e)
            print('ERROR')
        print('Finished at ',time.time())
        print("--- %s seconds ---" % (time.time() - start_time))
    
def main():
    data_provinsi='/data/ksa/00_Data_Input/bounding_box_gt.gpkg'
    gpd_provinsi=gpd.read_file(data_provinsi)
    for index, rows in gpd_provinsi.iterrows():
        print('########################################################')
        bound=rows.geometry
        kdkab=rows.idkab
        path_file = f'/data/ksa/01_Image_Acquisition/04_Json_Raw_Download/{kdkab}_groundtruth_metadata_ASF.json'
        results = asf.search(
            platform= asf.PLATFORM.SENTINEL1A,
            processingLevel=[asf.PRODUCT_TYPE.GRD_HD],
            start = date(2024, 9, 1),
            end = date(2024,9 , 30),
            intersectsWith = bound.wkt,
            )
        print(f'{kdkab}: Total Images Found: {len(results)}')
        #if os.path.exists(path_file):       
        #    len_corrupt_undowloded=docheck(path_file,kdkab)
        #    if len_corrupt_undowloded>0:
        #        dodownload(kdkab,bound)
        #else:
        #    dodownload(kdkab,bound)
        print('########################################################')

if __name__ == "__main__":
    main()