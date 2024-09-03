# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 16:43:37 2023

@author: Fatih Durmaz

One Hot Encoding
"""

import pandas as pd

df = pd.read_csv('test.csv')
print(df, end='\n\n')

from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse=False)

ohe.fit(df[['Renk Tercihi']])
ohe_data = ohe.transform(df[['Renk Tercihi']])
print(ohe_data, end='\n\n')

import numpy as np

cats = np.unique(df['Renk Tercihi'].to_numpy())
df.drop('Renk Tercihi', axis=1, inplace=True)
df[cats] = ohe_data
print(df, end='\n\n')