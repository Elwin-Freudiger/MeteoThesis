import pandas as pd
import numpy as np
from tqdm import tqdm



#varlist 
varlist = ['precip', 'East', 'North', 'moisture', 'pression', 'temperature']

def merge_sort(var):
    col = f'{var}_interpolated'
    if var == 'precip':
        df = pd.read_csv('data/interpolated/precip_interpolated.csv', dtype={col:float})
    else:
        df = pd.read_csv(f'data/interpol/{var}_interpolated.csv', dtype={col: float})
    df_filter = df.drop(columns=['east', 'north', 'altitude'])
    df_filter['time'] = df_filter['time'].str.replace('-|:|\\s{1}', '', regex=True)
    df_filter['time'] = df_filter['time'].str.slice(stop=12)
    df_filter[col] = df_filter[col].fillna(False)
    df_filter[col] = df_filter[col].apply(lambda x: 1 if x else 0)
    return df_filter

def main():
    merged_df = None

    for var in tqdm(varlist, desc='Processing'):
        df = merge_sort(var)

        if merged_df is None:
            merged_df = df
        else:
            merged_df = pd.merge(merged_df, df, on=['station', 'time'], how='outer')
    sorted_df = merged_df[['time','station',
                           'precip', 'East', 'North', 'moisture', 'pression', 'temperature', 
                           'precip_interpolated','East_interpolated','North_interpolated','moisture_interpolated','pression_interpolated','temperature_interpolated']]
    renamed_df = sorted_df.rename(columns={'pression': 'pressure', 'pression_interpolated': 'pressure_interpolated'})
    renamed_df.to_csv('data/clean/valais_clean.csv', index=False)
    print('Dataframes merged!')

if __name__ == "__main__":
    main()