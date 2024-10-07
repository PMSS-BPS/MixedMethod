import pickle
import polars as pl
from datetime import datetime, timedelta
import numpy as np
from whittaker_eilers import WhittakerSmoother
from glob import glob
from tqdm import tqdm
import concurrent.futures
import os
import sys

def generate_date_pairs(year):
    start_date = datetime(year, 1, 1)
    date_pairs = []
    while start_date.year == year:
        end_date = start_date + timedelta(days=11)
        if end_date.year != year:
            break
        date_pairs.append([start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')])
        start_date = end_date + timedelta(days=1)
    return date_pairs

def prepare_dates():
    list_date = []
    for year in [2021, 2022, 2023]:
        list_date.extend(generate_date_pairs(year))
    list_date = [f"{start.replace('-', '')}_{end.replace('-', '')}" for start, end in list_date]
    return list_date

def do_preparation(ls_pickle):
    with open(ls_pickle, 'rb') as file:
        dt_pkl = pickle.load(file)
    return pl.DataFrame(dt_pkl)  # Convert the loaded pandas DataFrame to a Polars DataFrame

def process_idpoint(j, dt_pkl, list_date, mgrs_map):
    # Filter using boolean masking
    u = dt_pkl.filter(pl.col('idpoint') == j).sort('periode')
    
    ls_date = pl.DataFrame({'periode': list_date})
    temp2 = ls_date.join(u, on='periode', how='left').fill_null(0)
    temp2 = temp2.with_columns([
        pl.lit(j).alias('idpoint'),
        pl.lit(mgrs_map).alias('MGRS'),
        (pl.col('Sigma0_VH_db') != 0).cast(pl.Int8).alias('weight')
    ])
    
    weight_sum = temp2['weight'].sum()
    total_count = temp2.shape[0]
    
    whittaker_smoother = WhittakerSmoother(lmbda=1, order=2, data_length=total_count, weights=temp2['weight'].to_numpy())
    temp2 = temp2.with_columns([
            pl.Series(whittaker_smoother.smooth(temp2['Sigma0_VH_db'].to_numpy())).alias('Sigma0_VH_db_interp'),
            pl.Series(whittaker_smoother.smooth(temp2['Sigma0_VV_db'].to_numpy())).alias('Sigma0_VV_db_interp'),
            pl.Series(temp2['weight']).alias('weight'),
            pl.Series(temp2['MGRS']).alias('MGRS'),
            pl.Series(temp2['idpoint']).alias('idpoint')
        ])
    return temp2
    
def main(kdprov):
    print('=================================================')
    print('Begin Imputation for Province: ', kdprov)
    pickle_prov = glob(f'/data/ksa/03_Sampling/data/{kdprov}/*.pkl')
    print('Found:', len(pickle_prov), 'data')
    list_date = prepare_dates()
    num_workers = 40  # Adjust this based on your system's capability
    
    for i in pickle_prov:
        mgrs = os.path.basename(i).replace('.pkl', '').replace('sampling_', '')
        output = f'/data/ksa/04_Data_Preprocessing/{kdprov}/01_imputation/{mgrs}_imputed_data.pkl'
        print('-------------------------------------------------------')
        if os.path.exists(output):
            print(f'File {output} has been created')
            print('Skip it')
        else:
            print('Start for:', i)
            temp = pl.DataFrame()
            dt_pkl = do_preparation(i)
            list_idpoint = dt_pkl['idpoint'].unique().to_list()
            temp_list = [temp]
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                results = list(tqdm(executor.map(process_idpoint, list_idpoint, [dt_pkl]*len(list_idpoint), [list_date]*len(list_idpoint), [mgrs]*len(list_idpoint)), total=len(list_idpoint)))
            
            temp = pl.concat([temp] + results)
            print('Finish Imputation. Follow the results by compiling data.')
            temp=temp.to_pandas()
            temp['Sigma0_VH_db_imputation']=temp.apply(lambda y: y['Sigma0_VH_db'] if y['weight']>0 else y['Sigma0_VH_db_interp'], axis=1)
            temp['Sigma0_VV_db_imputation']=temp.apply(lambda y: y['Sigma0_VV_db'] if y['weight']>0 else y['Sigma0_VV_db_interp'], axis=1)
            with open(output, 'wb') as file:
                pickle.dump(temp[['periode', 'idpoint', 'MGRS', 'weight', 'Sigma0_VH_db_imputation','Sigma0_VV_db_imputation']], file) 
        print('-------------------------------------------------------')
    print('Finish')
    print('=================================================')

if __name__ == "__main__":
    kdprov = sys.argv[1]
    main(kdprov)
