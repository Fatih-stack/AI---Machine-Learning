# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 16:19:28 2023

@author: Fatih Durmaz
# Missing Data and Label encoder
"""

import pandas as pd

df = pd.read_csv('melb_data.csv')

print(f'Toplam satır sayısı: {df.shape[0]}, Toplam sütun sayısı: {df.shape[1]}', end='\n\n')
missing_info = df.isna().sum()
print('Eksik verilerin sütunlara göre dağılımı', end='\n\n')
print(missing_info, end='\n\n')

missing_columns = [name for name in df.columns if df[name].isna().any()]
print('Eksik verilerin ilişkin olduğu sütunlar', end='\n\n')
print(missing_columns, end='\n\n')

missing_ratio = df.isna().sum().sum() / df.size
print('Eksik verilerin oranları', end='\n\n')
print(missing_ratio, end='\n\n')

total_missing_rows = df.isna().any(axis=1).sum()
print('Eksik veri içeren satırların sayısı', end='\n\n')
print(total_missing_rows, end='\n\n')

total_missing_rows_ratio = df.isna().any(axis=1).sum() / len(df)
print('Eksik veri içeren satırların oranı', end='\n\n')
print(total_missing_rows_ratio, end='\n\n')

import numpy as np

impute_val = np.round(df['Car'].mean())
df['Car'] = df['Car'].fillna(impute_val)    # eşdeğeri # df['Car'].fillna(impute_val, inplace=True)

impute_val = np.round(df['BuildingArea'].mean())
df['BuildingArea'] = df['BuildingArea'].fillna(impute_val)    # eşdeğeri # df['BuildingArea'].fillna(impute_val, inplace=True)

impute_val = df['YearBuilt'].median()
df['YearBuilt'] = df['YearBuilt'].fillna(impute_val)    # eşdeğeri # df['YearBuilt'].fillna(impute_val, inplace=True)

impute_val = df['CouncilArea'].mode()
df['CouncilArea'] = df['CouncilArea'].fillna(impute_val[0])    # eşdeğeri # df['BuildingArea'].fillna(impute_val, inplace=True)

from sklearn.impute import SimpleImputer

si = SimpleImputer(strategy='mean')

df[['Car', 'BuildingArea']] = np.round(si.fit_transform(df[['Car', 'BuildingArea']]))

"""
si.fit(df[['Car', 'BuildingArea']])
df[['Car', 'BuildingArea']] = np.round(si.transform(df[['Car', 'BuildingArea']]))
"""

si.set_params(strategy='most_frequent')
df[['YearBuilt', 'CouncilArea']] = si.fit_transform(df[['YearBuilt', 'CouncilArea']])

"""
si.fit(df[['YearBuilt', 'CouncilArea']])
df[['YearBuilt', 'CouncilArea']] = si.transform(df[['YearBuilt', 'CouncilArea']])
"""

# Categorize and label encode

df = pd.read_csv('test.csv')

def label_encode(df, colnames):
    for colname in colnames:
        labels = df[colname].unique()
        for index, label in enumerate(labels):
            df.loc[df[colname] == label, colname] = index
        
print(df, end='\n\n')        
label_encode(df, ['Renk Tercihi', 'Cinsiyet'])
print(df, end='\n\n')   

from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('test.csv')
print(df, end='\n\n')

le = LabelEncoder()

df['Renk Tercihi'] = le.fit_transform(df['Renk Tercihi'])
df['Cinsiyet'] = le.fit_transform(df['Cinsiyet'])
print(df)

le.fit(df['Renk Tercihi'])
result = le.inverse_transform(np.array([2, 1, 1, 1, 2, 2, 1, 0]))
print(result)