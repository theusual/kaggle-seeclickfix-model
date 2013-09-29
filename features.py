'''
Functions for hand crafting features
'''
__author__ = 'Bryan Gregory'
__email__ = 'bryan.gregory1@gmail.com'
__date__ = '09-28-2013'

from sklearn.feature_extraction import DictVectorizer
from sklearn import  preprocessing
from scipy.sparse import coo_matrix, hstack, vstack
import numpy as np
import pandas as pd
from datetime import datetime

def hours(df):
    #-----create hour only, and ~6 hour ranges----#
    df['created_time_hrs'] = [str(x.hour) for x in df['created_time']]
    df['created_time_range'] = '1800-0000'
    for idx in df.index:
        if df['created_time'][idx].hour in [0,1,2,3,4]:
            df['created_time_range'][idx] = '0000-0500'
        if df['created_time'][idx].hour in [5,6,7,8,9,10,11]:
            df['created_time_range'][idx] = '0500-1200'
        if df['created_time'][idx].hour in [12,13,14,15,16,17]:
            df['created_time_range'][idx] = '1200-1800'

def age(df):
    #---Calc age in days---#
    df['age'] = [(datetime.now() - x).days for x in df['created_date']]

def city(df):
    df['city'] = 'Chicago'
    for idx in df.index:
        if round(df['longitude'][idx],0) == -77.0:
            df['city'][idx]='Richmond'
        if round(df['longitude'][idx],0) == -122.0:
            df['city'][idx]='Oakland'
        if round(df['longitude'][idx],0) == -73.0:
            df['city'][idx]='New Haven'

def lat_long(df):
    df['long_rnd2'] = [str(round(x,2)) for x in df.longitude]
    df['lat_rnd2'] = [str(round(x,2)) for x in df.latitude]
    df['long_lat_rnd2'] = df['long_rnd2'] + df['lat_rnd2']

def dayofweek(df):
    df['dayofweek'] = [str(x.weekday()) for x in df['created_date']]

def month(df):
    df['month'] = [str(x.month) for x in df['created_date']]

def standardize(df,features):
    #---------------------------------------------------------------------
    #Standardize list of quant features (remove mean and scale to unit variance)
    #---------------------------------------------------------------------
    ###Remove the target if it exists, to keep it from getting included in the feature matrix
    if 'label' in df.keys():
        del df['label'];

    scaler = preprocessing.StandardScaler()
    if features == 'all':
        mtx = scaler.fit_transform(df.as_matrix())
    else:
        mtx = scaler.fit_transform(df.ix[:,features].as_matrix())
    return mtx

#TODO: flag issues created on holidays?  (shameless reuse of feature from past Kaggle contest)
def holidays(df,holidays_list):
    df['holiday_fg'] = 0
    df['holiday_word'] = ''
    for idx in df.index:
        for holiday in holidays_list:
            if holiday.lower() in df.text_all[idx].lower():
                df['holiday_fg'][idx] = 1
                df['holiday_word'][idx] = holiday

def description_length(df):
    #calc length of description
    df['description_len'] = [len(x) for x in df.description]