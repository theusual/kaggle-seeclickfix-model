'''
Functions for data loading, cleaning, and merging data
'''
__author__ = 'Bryan Gregory'
__email__ = 'bryan.gregory1@gmail.com'
__date__ = '09-27-2013'

import numpy as np
import json
import re
import datetime

#Clean the data of inconsistencies, bad date fields, bad data types, nested columns, etc.
def clean(df):
    #-----Parse time from date----#
    df['created_time_orig'] = df['created_time']
    df['created_date'] = [datetime.datetime.strptime(x[:10], "%Y-%m-%d") for x in df['created_time_orig']]
    df['created_time'] = [datetime.datetime.strptime(x[11:], "%H:%M:%S") for x in df['created_time_orig']]

    #---Fill missing text data with token to ensure the record is not skipped by NLP----#
    df['description'] = df.description.fillna('ZZZZZZ')

    #---Fill missing categorical data with NA---#
    df['tag_type'] = df.tag_type.fillna('NA')
    df['source'] = df.source.fillna('NA')

    #---recode to ASCII, which ignores any special non-linguistic characters ---#
    #df['text_body'] = [x.encode('ascii', 'ignore') for x in df.text_body]

    #---Clean problem characters---#
    df['summary'] = [x.replace('\n',' ') for x in df.summary]
    df['description'] = [x.replace('\n',' ') for x in df.description]
    df['summary'] = [x.replace('\r',' ') for x in df.summary]
    df['description'] = [x.replace('\r',' ') for x in df.description]

    #----Convert data types-----#

    #----Remove any unnecessary columns----#

    return df