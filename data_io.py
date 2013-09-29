'''
Functions for data IO
'''
__author__ = 'Bryan Gregory'
__email__ = 'bryan.gregory1@gmail.com'
__date__ = '09-06-2013'

import json
import csv
import gc
import pandas as pd
import time
from datetime import datetime
from sklearn.externals import joblib

#import JSON data into a dict
def load_json(file_path):
    return [json.loads(line) for line in open(file_path)]

#import delimited flat file into a list
def load_flatfile(file_path, delimiter=''):
    temp_array = []
    #if no delimiter is specified, try to use the built-in delimiter detection
    if delimiter == '':
        csv_reader = csv.reader(open(file_path))
    else:
        csv_reader = csv.reader(open(file_path),delimiter)
    for line in csv_reader:
        temp_array += line
    return temp_array #[line for line in csv_reader]

#import delimited flat file into a pandas dataframe
def load_flatfile_to_df(file_path, delimiter=''):
    #if no delimiter is specified, try to use the built-in delimiter detection
    if delimiter == '':
        return pd.read_csv(file_path)
    else:
        return pd.read_csv(file_path, delimiter)

def save_predictions(dfTest,clf_name,model_name,submission_no):
    timestamp = datetime.now().strftime("--%d-%m-%y_%H%M")
    filename = 'Submits/'+'Sub'+str(submission_no)+timestamp+'--'+model_name+'.csv'

    #---Perform any manual predictions cleanup that may be necessary---#

    #save predictions
    dfTest['predictions_'+clf_name] = [x[0] for x in dfTest['predictions_'+clf_name]]
    dfTest.ix[:,['id','predictions_'+clf_name]].to_csv(filename, index=False)
    print 'Submission file saved as ',filename

def save_predictions_benchmark(dfTest_Benchmark,benchmark_name,submission_no):
    timestamp = datetime.now().strftime("--%d-%m-%y_%H%M")
    filename = 'Submissions/'+'Submission'+submission_no+timestamp+'--'+benchmark_name+'.csv'

    #save predictions
    dfTest_Benchmark.ix[:,['RecommendationId','benchmark_'+benchmark_name]].to_csv(filename,cols=['RecommendationId','stars'], index=False)
    print 'Submission file saved as ',filename

def save_model(clf,clfname):
    timestamp = datetime.now().strftime("%d-%m-%y_%H%M")
    filename = 'Models/'+timestamp+'--'+clfname+'.joblib.pk1'
    joblib.dump(clf, filename, compress=9)
    print 'Model saved as ',filename

def load_model(filename):
    return joblib.load(filename)

def save_text_features( output_file, feature_names ):
	o_f = open( output_file, 'wb' )
	feature_names = "\n".join( feature_names )
	o_f.write( feature_names )