import pandas as pd
import numpy as np



#varlist 
varlist = ['precip', 'East', 'North', 'moisture', 'pression', 'temperature']

def merge_sort(var):
    df = pd.read_csv(f'data/interpolated/{var}_interpolated.csv')
    df[var] = df[var].fillna(df[f'{var}_interpolated'])
    df.drop([f'{var}_interpolated'])

    return df