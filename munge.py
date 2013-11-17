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
    #----Convert created_time to date object ----#
    df['created_time'] = [datetime.datetime.strptime(x, "%m/%d/%Y %H:%M") for x in df.created_time]

    #-----Parse time from date----#
    df['created_date'] = [x.date() for x in df.created_time]
    df['created_time'] = [x.time() for x in df.created_time]
    #df['created_date'] = [datetime.datetime.strptime(x[:10], "%Y-%m-%d") for x in df['created_time_orig']]
    #df['created_time'] = [datetime.datetime.strptime(x[11:], "%H:%M:%S") for x in df['created_time_orig']]
    df['month'] = [str(x.month) for x in df['created_date']]

    #---Fill missing text data with token to ensure the record is not skipped by NLP----#
    df['description'] = df.description.fillna('ZZZZZZ')

    #---Fill missing categorical data with NA---#
    df['tag_type'] = df.tag_type.fillna('NA')
    df['source'] = df.source.fillna('NA')

    #---recode to ASCII, which ignores any special non-linguistic characters ---#
    df['summary'] = [x.decode('iso-8859-1') for x in df.summary]
    df['description'] = [x.decode('iso-8859-1') for x in df.description]

    #---Clean problem characters in any text features---#
    df['summary'] = [x.replace('\n',' ') for x in df.summary]
    df['description'] = [x.replace('\n',' ') for x in df.description]
    df['summary'] = [x.replace('\r',' ') for x in df.summary]
    df['description'] = [x.replace('\r',' ') for x in df.description]
    df['summary'] = [x.replace('\\',' ') for x in df.summary]
    df['description'] = [x.replace('\\',' ') for x in df.description]
    df['summary'] = [x.replace('?',' ') for x in df.summary]
    df['description'] = [x.replace('?',' ') for x in df.description]

    #---Lower case any text features---#
    for idx in df.index:
        df.summary[idx] = df.summary[idx].lower()
        df.description[idx] = df.description[idx].lower()
        df.tag_type[idx] = df.tag_type[idx].lower()

    #---Refactor target variables (labels) to allow use of RMSE as error metric when training,
    #---even though RMSLE is the competition's error metric---#
    if 'num_votes' in df.keys():
        df['num_votes_orig'] = df['num_votes']
        df['num_votes'] = np.log(df['num_votes_orig'] + 1)
        df['num_comments_orig'] = df['num_comments']
        df['num_comments'] = np.log(df['num_comments_orig'] + 1)
        df['num_views_orig'] = df['num_views']
        df['num_views'] = np.log(df['num_views_orig'] + 1)

    #----Convert data types-----#

    #----Remove any unnecessary columns----#

    #----Reduce training dataset by parsing off sections of irrelevant training records----#
    #Because of API changes (remote_api) use only months: 11/2012, 12/2012, 01/2013, 02/2013, 04/2013
    if 'num_votes' in df.keys():
        df = df[df.created_date > datetime.date(2012, 11, 1)]
        #df = df[df.month != '4']
    return df

def temporal_split(dfTrn, temporal_cutoff):
    dfTest = dfTrn[dfTrn.created_date >= datetime.date(temporal_cutoff[0],temporal_cutoff[1],temporal_cutoff[2])]
    dfTrn = dfTrn[dfTrn.created_date < datetime.date(temporal_cutoff[0],temporal_cutoff[1],temporal_cutoff[2])]
    return dfTrn,dfTest