'''
Kaggle+Hackathon!

TODO:
-DONE: Clean duplicate issues and other noise
-DONE: Create age in days
-DONE: City
-DONE: NLP on summary, NLP on description
-DONE: dictvectorizer on tag_type
-DONE: dictvectorizer on source
-DONE: Round long/lat to hundredths and create vector
-DONE: Create day of week as binary vector
-Add feature for length of description
-Break apart Sum+Descr into 2 feature vectors
-Data subsets for each city
-long/lat neighborhood mapping
-Sparse NN on non text featurse
-libFM on non text features
-Ensemble of text+nontext models
-accomodate for age of record somehow?
'''
__author__ = 'Bryan Gregory'
__email__ = 'bryan.gregory1@gmail.com'
__date__ = '09-27-2013'

import munge
import train
import data_io
import features

from scipy import sparse
from sklearn.externals import joblib
import sys
import csv
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.datasets import dump_svmlight_file
from sklearn.feature_extraction import text, DictVectorizer
from sklearn import (metrics, cross_validation, linear_model, ensemble, preprocessing, svm, neighbors, gaussian_process, naive_bayes, neural_network, pipeline, lda)

from scipy.sparse import coo_matrix, hstack, vstack


#def main():
#------------------------------------------------#
#-------Data Loading/Cleaning/Munging------------#
#------------------------------------------------#

dfTrn = data_io.load_flatfile_to_df("Data/train.csv")
dfTest = data_io.load_flatfile_to_df("Data/test.csv")

#--Clean the data--#
dfTrn = munge.clean(dfTrn)
dfTest = munge.clean(dfTest)

#------------------------------------------------#
#-------Feature creation-------------------------#
#------------------------------------------------#

#---Add hand crafted features---#
features.hours(dfTrn)
features.hours(dfTest)

features.age(dfTrn)
features.age(dfTest)

features.city(dfTrn)
features.city(dfTest)

features.lat_long(dfTrn)
features.lat_long(dfTest)

features.dayofweek(dfTrn)
features.dayofweek(dfTest)

features.month(dfTrn)
features.month(dfTest)

#-------------------------------------------------------#
#----Create data subsets--------------------------------#
#-------------------------------------------------------#
#----Skip this section if modeling on whole data set----#
#-------------------------------------------------------#


#---Create subset of just one city for vectorizing, training, and testing---#
#make copy of original whole dataset
dfTrn_All = dfTrn.ix[:]
dfTest_All = dfTest.ix[:]

#Chicago
dfTrn = dfTrn_All[dfTrn_All.city == 'Chicago']
dfTest = dfTest_All[dfTest_All.city == 'Chicago']

#Richmond
dfTrn = dfTrn_All[dfTrn_All.city == 'Richmond']
dfTest = dfTest_All[dfTest_All.city == 'Richmond']

#Oakland
dfTrn = dfTrn_All[dfTrn_All.city == 'Oakland']
dfTest = dfTest_All[dfTest_All.city == 'Oakland']

#New Haven
dfTrn = dfTrn_All[dfTrn_All.city == 'New Haven']
dfTest = dfTest_All[dfTest_All.city == 'New Haven']


#-------------------------------------------------#
#----Create stand-alone feature vectors-----------#
#-------------------------------------------------#

#---Create bag of words vector from summary--#
#count_vec = text.CountVectorizer(min_df = 3, max_df = 0.9, strip_accents = 'unicode', binary = True)
#count_vec.fit(np.append(dfTrn.summary.values,dfTest.summary.values))
#mtxTrn_Text = count_vec.transform(dfTrn.summary.values)
#mtxTest_Text = count_vec.transform(dfTest.summary.values)

#---Create tfid vector from summary---#
#Word:
tfidf_vec = text.TfidfVectorizer(min_df=3,  max_features=None, strip_accents='unicode',analyzer='word', token_pattern=r'\w{1,}',ngram_range=(1, 2), use_idf=1,smooth_idf=1,sublinear_tf=1)
#Char_WB:
#tfidf_vec = text.TfidfVectorizer(min_df=3,  max_features=None, strip_accents='unicode',analyzer='char_wb', token_pattern=r'\w{1,}',ngram_range=(2, 30), use_idf=1,smooth_idf=1,sublinear_tf=1)
tfidf_vec.fit(np.append(dfTrn.summary.values,dfTest.summary.values))
mtxTrn_Text = tfidf_vec.transform((dfTrn.summary.values))
mtxTest_Text = tfidf_vec.transform((dfTest.summary.values))

#---Create tfid vector from description+summary---#
#Word:
#tfidf_vec = text.TfidfVectorizer(min_df=3,  max_features=None, strip_accents='unicode',analyzer='word', token_pattern=r'\w{1,}',ngram_range=(1, 2), use_idf=1,smooth_idf=1,sublinear_tf=1)
#Char_WB:
tfidf_vec = text.TfidfVectorizer(min_df=3,  max_features=None, strip_accents='unicode',analyzer='char_wb', token_pattern=r'\w{1,}',ngram_range=(2, 30), use_idf=1,smooth_idf=1,sublinear_tf=1)
tfidf_vec.fit(np.append(np.append(dfTrn.summary.values,dfTrn.description.values),np.append(dfTest.summary.values,dfTest.description.values)))
mtxTrn_Text = tfidf_vec.transform(dfTrn.summary.values+dfTrn.description.values)
mtxTest_Text = tfidf_vec.transform(dfTest.summary.values+dfTest.description.values)

#---Vectorize cities--#
city_vec = DictVectorizer().fit([{'feature':value} for value in np.append(dfTrn.city,dfTest.city)])
mtxTrn_City= city_vec.transform([{'feature':value} for value in dfTrn.city])
mtxTest_City = city_vec.transform([{'feature':value} for value in dfTest.city])

#---Vectorize tag type--#
tagtype_vec = DictVectorizer().fit([{'feature':value} for value in np.append(dfTrn.tag_type,dfTest.tag_type)])
mtxTrn_Tagtype= tagtype_vec.transform([{'feature':value} for value in dfTrn.tag_type])
mtxTest_Tagtype = tagtype_vec.transform([{'feature':value} for value in dfTest.tag_type])

#---Vectorize source--#
source_vec = DictVectorizer().fit([{'feature':value} for value in np.append(dfTrn.source,dfTest.source)])
mtxTrn_Source= source_vec.transform([{'feature':value} for value in dfTrn.source])
mtxTest_Source = source_vec.transform([{'feature':value} for value in dfTest.source])

#---Vectorize Hours-#
hrs_vec = DictVectorizer().fit([{'feature':value} for value in np.append(dfTrn.created_time_hrs,dfTest.created_time_hrs)])
mtxTrn_Hrs= source_vec.transform([{'feature':value} for value in dfTrn.created_time_hrs])
mtxTest_Hrs = source_vec.transform([{'feature':value} for value in dfTest.created_time_hrs])

#---Vectorize Hours Range-#
hrsrange_vec = DictVectorizer().fit([{'feature':value} for value in np.append(dfTrn.created_time_range,dfTest.created_time_range)])
mtxTrn_HrsRange= hrsrange_vec.transform([{'feature':value} for value in dfTrn.created_time_range])
mtxTest_HrsRange = hrsrange_vec.transform([{'feature':value} for value in dfTest.created_time_range])

#---Vectorize Day Of Week ---#
dayofweek_vec = DictVectorizer().fit([{'feature':value} for value in np.append(dfTrn.dayofweek,dfTest.dayofweek)])
mtxTrn_dayofweek= dayofweek_vec.transform([{'feature':value} for value in dfTrn.dayofweek])
mtxTest_dayofweek = dayofweek_vec.transform([{'feature':value} for value in dfTest.dayofweek])

#---Vectorize Long+Lat ---#
longlat_vec = DictVectorizer().fit([{'feature':value} for value in np.append(dfTrn.long_lat_rnd2,dfTest.long_lat_rnd2)])
mtxTrn_longlat= longlat_vec.transform([{'feature':value} for value in dfTrn.long_lat_rnd2])
mtxTest_longlat = longlat_vec.transform([{'feature':value} for value in dfTest.long_lat_rnd2])


#-------------------------------------------------#
#--- Machine Learning (finally the good stuff!----#
#-------------------------------------------------#

#--Select classifier--#

#clf = linear_model.LogisticRegression(penalty='l2', dual=True, tol=0.0001, C=1, fit_intercept=True, intercept_scaling=1.0, class_weight=None, random_state=None);clf_name='log'
#clf = ensemble.RandomForestRegressor(n_estimators=50);  clfname='RFReg_50'
#clf = ensemble.ExtraTreesRegressor(n_estimators=30)  #n_jobs = -1 if running in a main() loop
#clf = linear_model.SGDRegressor(alpha=0.001, n_iter=800,shuffle=True); clf_name='SGD_001_800'
#clf = gaussian_process.GaussianProcess(corr='cubic', theta0=1e-2, thetaL=1e-4, thetaU=1e-1, random_start=100)   #doesn't work on large datasets
clf = linear_model.Ridge();clf_name = 'RidgeReg'
#clf = linear_model.LinearRegression();clf_name = 'LinReg'
#clf = linear_model.ElasticNet()
#clf = linear_model.Lasso();clf_name = 'Lasso'
#clf = linear_model.LassoCV(cv=3);clf_name = 'LassoCV'
#clf = svm.SVR(kernel = 'linear',cache_size = 6000.0) #use .ravel(), kernel='rbf','linear'

#---select, scale, and combine features to use---#
#Select quant features.  ----  Unused quant_features : []
quant_features = ['age']  # ['fm_preds']
dfTrn_ML=dfTrn;dfTest_ML= dfTest;

mtxTrn=mtxTrn_Text;mtxTest=mtxTest_Text
#mtxTrn=mtxTrn_City; mtxTest=mtxTest_City
#mtxTrn=mtxTrn_Tagtype; mtxTest=mtxTest_Tagtype
#mtxTrn=mtxTrn_Source; mtxTest=mtxTest_Source
#mtxTrn=mtxTrn_HrsRange; mtxTest=mtxTest_HrsRange
#mtxTrn=mtxTrn_Hrs; mtxTest=mtxTest_Hrs
#mtxTrn=mtxTrn_longlat; mtxTest=mtxTest_longlat
#mtxTrn=mtxTrn_dayofweek; mtxTest=mtxTest_dayofweek

#Standardize/scale as needed, if not needed just convert feature to matrix
mtxTrn = features.standardize(dfTrn_ML,quant_features)
mtxTest = features.standardize(dfTest_ML,quant_features)

#Combine the quant features and the vectorized features
###For num_views (load word sum to mtx_Text first):
mtxTrn = hstack([mtxTrn_Source,mtxTrn_Text])  #mtxTrn_City,mtxTrn_Tagtype,mtxTrn_Source,mtxTrn_HrsRange
mtxTest = hstack([mtxTest_Source,mtxTest_Text]) #mtxTest_City,mtxTest_Tagtype,mtxTest_Source,mtxTest_HrsRange

###For num_comments (load word sum to mtx_Text first):
mtxTrn = hstack([mtxTrn_Source,mtxTrn_Text,mtxTrn_longlat])  #mtxTrn_City,mtxTrn_Tagtype,mtxTrn_Source,mtxTrn_HrsRange
mtxTest = hstack([mtxTest_Source,mtxTest_Text,mtxTest_longlat]) #mtxTest_City,mtxTest_Tagtype,mtxTest_Source,mtxTest_HrsRange

###For num_votes (load word sum to mtx_Text first):
mtxTrn = hstack([mtxTrn_Source,mtxTrn_Text,mtxTrn_City, mtxTrn_longlat])  #mtxTrn_City,mtxTrn_Tagtype,mtxTrn_Source,mtxTrn_HrsRange
mtxTest = hstack([mtxTest_Source,mtxTest_Text,mtxTest_City, mtxTest_longlat]) #mtxTest_City,mtxTest_Tagtype,mtxTest_Source,mtxTest_HrsRange

#---Select target---#
mtxTarget = dfTrn.ix[:,['num_votes']].as_matrix()
mtxTarget = dfTrn.ix[:,['num_comments']].as_matrix()
mtxTarget = dfTrn.ix[:,['num_views']].as_matrix()

#---Cross Validation---#
cv_preds = train.cross_validate(hstack([sparse.csr_matrix(dfTrn.urlid.values).transpose(),mtxTrn]),mtxTarget.ravel(),folds=10,SEED=42,test_size=.1,clf=clf,clf_name=clf_name,pred_fg=True)  #may require mtxTrn.toarray()
train.cross_validate(mtxTrn,mtxTarget.ravel(),folds=10,SEED=42,test_size=.1,clf=clf,clf_name=clf_name,pred_fg=False)  #may require mtxTrn.toarray()
train.cross_validate_using_benchmark('global_mean',dfTrn, mtxTrn,mtxTarget,folds=20)

#---Calculate the degree of variance between ground truth and the mean of the CV predictions.----#
#---Returns a list of all training records with their average variance---#
train.calc_cv_preds_var(dfTrn,cv_preds)

#--Use classifier for predictions--#
dfTest, clf = train.predict(mtxTrn,mtxTarget,mtxTest,dfTest,clf,clf_name) #may require mtxTest.toarray()

#--Save predictions to file--#
data_io.save_predictions(dfTest,clf_name,'_Votes_Chi_Ridge',submission_no)
data_io.save_predictions(dfTest,clf_name,'_Comments_Chi_Ridge',submission_no)
data_io.save_predictions(dfTest,clf_name,'_Views_Chi_Ridge',submission_no)

#------------------------------------------------#
#-------Optional/Misc.---------------------------#
#------------------------------------------------#
#--Save feature matrices in svm format for external modeling--#
y_trn = np.asarray(dfTrn.num_votes)
y_test = np.ones(mtxTest.shape[0], dtype = int )
dump_svmlight_file(mtxTrn, y_trn, f = "Data/Votes_trn.svm", zero_based = False )
dump_svmlight_file(mtxTest, y_test, f = "Data/Votes_test.svm", zero_based = False )

y_trn = np.asarray(dfTrn.num_comments)
y_test = np.ones(mtxTest.shape[0], dtype = int )
dump_svmlight_file(mtxTrn, y_trn, f = "Data/Comments_trn.svm", zero_based = False )
dump_svmlight_file(mtxTest, y_test, f = "Data/Comments_test.svm", zero_based = False )

y_trn = np.asarray(dfTrn.num_views)
y_test = np.ones(mtxTest.shape[0], dtype = int )
dump_svmlight_file(mtxTrn, y_trn, f = "Data/Views_trn.svm", zero_based = False )
dump_svmlight_file(mtxTest, y_test, f = "Data/Views_test.svm", zero_based = False )

#--Save a model to joblib file--#
data_io.save_model(clf,'rf_500_TextAll')

#--Load a model from joblib file--#
data_io.load_model('Models/040513--rf_500_TextAll.joblib.pk1')

#--Save text feature names list for later reference--#
data_io.save_text_features("Data/text_url_features.txt",tfidf_vec.get_feature_names())

#--Save a dataframe to CSV--#
filename = 'Data/Test_Holidays'+'.csv'
###Fill missing values with 0 to preserve all columns for csv export
dfTest.ix[:,['holiday_fg','holiday_word','urlid']].to_csv(filename, index=True,  sep='\t',encoding='utf-8')