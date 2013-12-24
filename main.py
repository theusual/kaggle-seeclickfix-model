__author__ = 'Bryan Gregory'
__email__ = 'bryan.gregory1@gmail.com'
__date__ = '12-03-2013'

#internal modules
import munge
import train
import data_io
import features
import ensembles

#external modules
from scipy import sparse
from sklearn.externals import joblib
import sys
import csv
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.datasets import dump_svmlight_file
from sklearn import (metrics, cross_validation, linear_model, ensemble, tree, preprocessing, svm, neighbors, gaussian_process, naive_bayes, neural_network, pipeline, lda)

class Model(object):
    #Class attributes:
    #model_name = descriptive name of model
    #target = target variable for this model
    #segment = segment of data for this model
    #classifier_name = descriptive name of estimator to use
    #classifer = the sklearn estimator to use
    #features = a dictionary of features to use in model. The key is feature name, the value is an binary array with
    #   actual values of feature for train set and actual values of feature for test set.
    #postprocess_scalar = the scalar to apply to all predictions generated from model
    def __init__(self, model_name, target, segment, classifier_name, features, postprocess_scalar):
        self.model_name = model_name
        self.target = target
        self.segment = segment
        self.classifier_name = classifier_name
        self.set_classifier(classifier_name)
        self.set_features(features)
        self.postprocess_scalar = round(np.float32(postprocess_scalar), 4)
    def set_classifier(self, classifier):
        if classifier[:3] == 'GBM':
            self.classifier = ensemble.GradientBoostingRegressor(max_depth=1, random_state=888, loss='ls')
            self.classifier.n_estimators = np.float32(classifier[4:classifier[4:].find("_")+4])
            self.classifier.learning_rate = round(np.float32(classifier[8:]), 4)
        if classifier[:3] == 'SGD':
            self.classifier = linear_model.SGDRegressor(shuffle=True)
            self.classifier.n_iter = np.float32(classifier[4:classifier[4:].find("_")+4])
            self.classifier.alpha = round(np.float32(classifier[8:]), 4)
            self.classifier.random_state = 8888
            self.classifier.alpha = .0001
        if classifier[:3] == 'ADA':
            self.classifier = ensemble.AdaBoostRegressor()
            self.classifier.n_iter = np.float32(classifier[4:classifier[4:].find("_")+4])
            self.classifier.alpha = round(np.float32(classifier[8:]), 4)
            self.classifier.random_state = 8888
        if classifier[:7] == 'LASSOCV':
            self.classifier = linear_model.LassoCV()
            self.classifier.cv = np.int(classifier[8:])
    def set_features(self, features):
        #Accepts a list of features
        self.features = dict((feature,['','']) for feature in features)
    def predict(self,dfTrn,dfTest):
        #Segment the train and test data to match current model
        dfTrn, dfTest = munge.segment_data(dfTrn, dfTest, self.segment)
        #Vectorize each text, categorical, or boolean feature into a train and test matrix stored in model.features
        features.vectors(dfTrn, dfTest, self.features)
        #Transform or scale any numerical features and create feature vector
        features.numerical(dfTrn, dfTest, self.features)
        #Make predictions
        mtxTrn, mtxTest, mtxTrnTarget, mtxTestTarget = train.combine_features(self, dfTrn, dfTest)
        train.predict(mtxTrn,mtxTrnTarget.ravel(),mtxTest,dfTest,self)
        #Store predictions in dataframe as class attribute
        self.dfPredictions = dfTest.ix[:,['id',self.target]]
        #Export predictions (optional)
        if settings["export_predictions_each_model"] == 'true':
            data_io.save_predictions(dfTest,self,'test')

#def main():
#-------------Load Settings/Models----------------------#
#Get environment settings
settings = json.loads(open("SETTINGS.json").read())

#Get model settings
model_settings = json.loads(open("MODELS.json").read())

#TODO: If settings are use cached model, then skip initializing and load models instead
#Initialize list of model classes
models = []
for model in model_settings:
    new_model = Model(model_name=model,target=model_settings[model]['target'],segment=model_settings[model]['segment'],
                      classifier_name=model_settings[model]['classifier_name'],features=model_settings[model]['features'],
                      postprocess_scalar=model_settings[model]['postprocess_scalar'])
    models.append(new_model)

#Manually add a new model:
#models.append(Model(model_name="Weak Descr Len",target="num_views",segment="Richmond",
#                   classifier_name="RF",features=["description_length"],postprocess_scalar=1))

#-------Data Loading/Cleaning/Munging------------#
#Load the data
dfTrn = data_io.load_flatfile_to_df(settings['filename_train'])
dfTest = data_io.load_flatfile_to_df(settings['filename_test'])
dfCV = data_io.load_flatfile_to_df('Data/CV.csv')

#Clean/Munge the data
dfTrn = munge.clean(dfTrn)
dfTest = munge.clean(dfTest)


#-------Feature creation-------------------------#
#Add all currently used hand crafted features to dataframes
features.add(dfTrn)
features.add(dfTest)

#---------Data slicing/parsing--------------------------#
#Temporal split of data for CV
if settings['generate_cv_score'] == 'true' and settings['cv_method'] == 'april':
    dfTrnCV, dfTestCV = munge.temporal_split(dfTrn, (2013, 04, 1))
elif settings['generate_cv_score'] == 'true' and settings['cv_method'] == 'march':
    #take an addtional week from February b/c of lack of remote_api source issues in March
    dfTrnCV, dfTestCV = munge.temporal_split(dfTrn, (2013, 02, 21))
elif settings['generate_cv_score'] == 'true' and settings['cv_method'] == 'list_split':
    #load stored list of data points and use those for CV
    dfCVlist = pd.DataFrame({'id': data_io.load_cached_object("Cache/cv_issue_ids.pkl"), 'dummy': 0})
    dfTrnCV, dfTestCV = munge.list_split(dfTrn, dfCVlist)

features.sub_feature(dfTrnCV,'zipcode','neighborhood',["Richmond","Oakland","Manchester","Chicago","New Haven"])
features.sub_feature(dfTestCV,'zipcode','neighborhood',["Richmond","Oakland","Manchester","Chicago","New Haven"])
features.knn_thresholding(dfTrnCV,'neighborhood',6)
features.knn_thresholding(dfTestCV,'neighborhood',6)

#--------------Modelling-------------------------------#
#Iterate through every model, first segmenting the data for that model, then creating the feature values for that model,
#then training the model, scoring the model in CV (optional), exporting the model to a joblib file (optional),
#and finally saving predictions (optional).  Options chosen are dependent on SETTINGS.json.
for model in models[12:]:
    features_list = (map(str,model.features.keys()))
    features_list.sort()
    print "=============================================================================================================="
    print "=============================================================================================================="
    print "=============================================================================================================="
    print "MODEL: %s    SEGMENT: %s     TARGET: %s "  % (model.model_name, model.segment, model.target)
    print "FEATURES: %s" % features_list
    print "LEARNING ALGORITHM: %s    POST-PROCESS SCALAR: %s " % (model.classifier_name,model.postprocess_scalar)
    print "=============================================================================================================="
    if settings['generate_cv_score'] == 'true':
        dfTrn_Segment, dfTest_Segment = munge.segment_data(dfTrnCV, dfTestCV, model.segment)
        #Vectorize each text, categorical, or boolean feature into a train and test matrix stored in model.features
        features.vectors(dfTrn_Segment, dfTest_Segment, model.features)
        #Transform or scale any numerical features and create feature vector
        features.numerical(dfTrn_Segment, dfTest_Segment, model.features)
        #Generate predictions
        train.cross_validate(model, settings, dfTrn_Segment, dfTest_Segment)
        #Output each model's CV predictions to file (optional)
        if settings['export_cv_predictions_each_model'] == 'true':
            #Output individual model predictions
            data_io.save_predictions(dfTest_Segment,model,'CV_list')
        if settings['export_cv_predictions_total'] == 'true':
            if 'dfCVPredictions' not in locals():
                dfCVPredictions = pd.DataFrame(columns=['id'])
            #If current model's predictions are for new ID's, then append the new ID's.  Otherwise merge the predictions.
            if len(dfCVPredictions.merge(model.dfPredictions,how='inner',on='id')) == 0:
                dfTestPredictions = dfTestPredictions.append(model.dfPredictions)
            elif model.target in dfTestPredictions.columns:
                try:
                    dfTestPredictions[model.target] = pd.concat([dfTestPredictions[model.target].dropna(),
                                     model.dfPredictions[model.target]]).reindex_like(dfTestPredictions)
                except AssertionError:
                    dfTestPredictions[model.target] = pd.concat([dfTestPredictions[model.target].dropna(),
                                     model.dfPredictions[model.target]], ignore_index=True).reindex_like(dfTestPredictions)
            else:
                dfTestPredictions = dfTestPredictions.merge(model.dfPredictions,how='left',on='id')
    #If there are no existing predictions already calculated AND predictions are needed then make predictions
    if settings['export_predictions_each_model'] == 'true' or settings['export_predictions_total'] == 'true':
        if not hasattr(model,'dfPredictions'):
            model.predict(dfTrn, dfTest)
        else:
            print 'Predictions found.  Using cached predictions.'
    #Merge each model's predictions with all test data for later export (optional)
    if settings['export_predictions_total'] == 'true':
        if 'dfTestPredictions' not in locals():
            dfTestPredictions = pd.DataFrame(columns=['id'])
        #If current model's predictions are for new ID's, then append the new ID's.  Otherwise merge the predictions.
        if len(dfTestPredictions.merge(model.dfPredictions,how='inner',on='id')) == 0:
            dfTestPredictions = dfTestPredictions.append(model.dfPredictions)
        elif model.target in dfTestPredictions.columns:
            try:
                dfTestPredictions[model.target] = pd.concat([dfTestPredictions[model.target].dropna(),
                                 model.dfPredictions[model.target]]).reindex_like(dfTestPredictions)
            except AssertionError:
                dfTestPredictions[model.target] = pd.concat([dfTestPredictions[model.target].dropna(),
                                 model.dfPredictions[model.target]], ignore_index=True).reindex_like(dfTestPredictions)
        else:
            dfTestPredictions = dfTestPredictions.merge(model.dfPredictions,how='left',on='id')


#---Ensemble Averaging----#
reload(ensembles);ensemble_CV = ensembles.EnsembleAvg(targets=targets,id="id")
ensemble_CV.load_models_csv(filepath="Submits/BryanModel-Updated-CV.csv")
ensemble_CV.load_models_csv(filepath="Submits/ridge_38_cv.csv")
ensemble_CV.load_models_csv(filepath="Submits/weak_geo_cv.csv")
#Parse segments
ensemble_CV.sub_models_segment.append\
        (ensemble_CV.sub_models[0][ensemble_CV.sub_models[0]['Segment'] == "Richmond"].reset_index())
ensemble_CV.sub_models_segment.append\
        (ensemble_CV.sub_models[1][ensemble_CV.sub_models[1]['Segment'] == "Richmond"].reset_index())
ensemble_CV.sub_models_segment.append\
        (ensemble_CV.sub_models[2][ensemble_CV.sub_models[2]['Segment'] == "Richmond"].reset_index())
dfSegTestCV = dfTestCV.merge(ensemble_CV.sub_models_segment[0].ix[:,['id']],on='id',how="inner")
#set targets
ensemble_CV.targets=["num_views"]
#Transform CV targets back to normal
for target in ensemble_CV.targets:
    dfSegTestCV[target]=np.exp(dfSegTestCV[target])-1
#Load groundtruth values for CV
ensemble_CV.load_df_true_segment(dfSegTestCV)
#Sort all dataframes by ID for easy comparison
ensemble_CV.sort_dataframes("id")
#Transform predictions to log space for averaging
ensemble_CV.transform_targets_log()
#Set weights
#Remote_API: weights = [{"num_views":.16,"num_votes":.3,"num_comments":.9},{"num_views":.84,"num_votes":.7,"num_comments":.1}]
#Richmond:   weights = [{"num_views":.7,"num_votes":.45,"num_comments":.7},{"num_views":.3,"num_votes":.55,"num_comments":.3},{"num_views":.4"}]
#Oakland weights = [{"num_views":.2,"num_votes":.1,"num_comments":.7},{"num_views":.8,"num_votes":.9,"num_comments":.3}]
weights = [{"num_views":.2,"num_votes":.1,"num_comments":.6},{"num_views":.8,"num_votes":.9,"num_comments":.4}]
#Create ensemble average
#ensemble_CV.create_ensemble([0,1],weights)
ensemble_CV.create_ensemble_segment([0,1,2],weights)
#Score the ensemble
#ensemble_CV.score_rmsle(ensemble_CV.sub_models_segment[0], df_true=ensemble_CV.df_true_segment)
ensemble_CV.score_rmsle(ensemble_CV.df_ensemble_segment, df_true=ensemble_CV.df_true_segment)


#---Use regressor to find ideal weights for ensemble---#
for target_label in ensemble_CV.targets:
    clf.fit_intercept=False
    train = np.hstack((ensemble_CV.sub_models_segment[0].ix[:,[target_label]].as_matrix(),
                       ensemble_CV.sub_models_segment[1].ix[:,[target_label]].as_matrix(),
                       ensemble_CV.sub_models_segment[2].ix[:,[target_label]].as_matrix()))
    target = ensemble_CV.df_true_segment.ix[:,[target_label]].as_matrix()
    clf.fit(train,target)
    try:
        for i in range(len(ensemble_CV.sub_models_segment)):
            weights[i][target_label]=clf.coef_[i]
    except:
        for i in range(len(ensemble_CV.sub_models_segment)):
            weights[i][target_label]=clf.coef_[0][i]
    print clf.coef_

#-----------Test Ensemble--------#
reload(ensembles);ensemble_CV = ensembles.EnsembleAvg(targets=["num_views"],id="id")
ensemble_test.load_models_csv(filepath="Submits/BryanModel-Updated.csv")
ensemble_test.load_models_csv(filepath="Submits/ridge_38_test.csv")
ensemble_test.load_models_csv(filepath="Submits/weak_geo_svr_.75.csv")
#Parse segments
ensemble_test.sub_models_segment.append\
        (ensemble_test.sub_models[0][ensemble_CV.sub_models[0]['Segment'] == "Richmond"].reset_index())
ensemble_test.sub_models_segment.append\
        (ensemble_test.sub_models[1][ensemble_CV.sub_models[1]['Segment'] == "Richmond"].reset_index())
ensemble_test.sub_models_segment.append\
        (ensemble_test.sub_models[2][ensemble_CV.sub_models[2]['Segment'] == "Richmond"].reset_index())
dfSegTestCV = dfTestCV.merge(ensemble_CV.sub_models_segment[0].ix[:,['id']],on='id',how="inner")

#Transform CV targets back to normal
for target in ensemble_CV.targets:
    dfSegTestCV[target]=np.exp(dfSegTestCV[target])-1
#Load groundtruth values for CV
ensemble_CV.load_df_true_segment(dfSegTestCV)
#Sort all dataframes by ID for easy comparison
ensemble_CV.sort_dataframes("id")
#Transform predictions to log space for averaging
ensemble_CV.transform_targets_log()
#Set weights
#Remote_API: weights = [{"num_views":.16,"num_votes":.3,"num_comments":.9},{"num_views":.84,"num_votes":.7,"num_comments":.1}]
#Richmond:   weights = [{"num_views":.7,"num_votes":.45,"num_comments":.7},{"num_views":.3,"num_votes":.55,"num_comments":.3},{"num_views":.4"}]
#Oakland weights = [{"num_views":.2,"num_votes":.1,"num_comments":.7},{"num_views":.8,"num_votes":.9,"num_comments":.3}]
weights = [{"num_views":.2,"num_votes":.1,"num_comments":.6},{"num_views":.8,"num_votes":.9,"num_comments":.4}]
#Create ensemble average
#ensemble_CV.create_ensemble([0,1],weights)
ensemble_CV.create_ensemble_segment([0,1,2],weights)
#Score the ensemble
#ensemble_CV.score_rmsle(ensemble_CV.sub_models_segment[0], df_true=ensemble_CV.df_true_segment)
ensemble_CV.score_rmsle(ensemble_CV.df_ensemble_segment, df_true=ensemble_CV.df_true_segment)



#Clean off outliers
#Views
dfTrn = dfTrn[dfTrn.num_views_orig  < 3]
#dfTest = dfTest[dfTest.num_views_orig < 3]


#----------------------------------------------------------------------------#
#----List of all current SKLearn learning algorithms capable of regression---#
#----------------------------------------------------------------------------#

#----For larger data sets------#
#clf = linear_model.LogisticRegression(penalty='l2', dual=True, tol=0.0001, C=1, fit_intercept=True, intercept_scaling=1.0, class_weight=None, random_state=None);clf_name='log'
#clf = linear_model.SGDRegressor(alpha=0.001, n_iter=800,shuffle=True); clf_name='SGD_001_800'
#clf = linear_model.Ridge();clf_name = 'RidgeReg'
clf = linear_model.LinearRegression();clf_name = 'LinReg'
#clf = linear_model.ElasticNet()
#clf = linear_model.Lasso();clf_name = 'Lasso'
#clf = linear_model.LassoCV(cv=3);clf_name = 'LassoCV'
#clf = svm.SVR(kernel = 'poly',cache_size = 16000.0) #use .ravel(), kernel='rbf','linear','poly','sigmoid'
#clf = svm.NuSVR(nu=0.5, C=1.0, kernel='linear', degree=3, gamma=0.0, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=20000, verbose=False, max_iter=-1)

#----For smaller data sets------#   (Do not work or have very long training times on large sparse datasets)  Require .todense()
#clf = ensemble.RandomForestRegressor(n_estimators=50);  clfname='RFReg_50'
#clf = ensemble.ExtraTreesRegressor(n_estimators=30)  #n_jobs = -1 if running in a main() loop
#clf = ensemble.GradientBoostingRegressor(n_estimators=700, learning_rate=.1, max_depth=1, random_state=888, loss='ls');clf_name='GBM'
clf = ensemble.AdaBoostRegressor(base_estimator=tree.DecisionTreeRegressor(compute_importances=None, criterion='mse', max_depth=3,
                                                                                   max_features=None, min_density=None, min_samples_leaf=1,
                                                                                   min_samples_split=2, random_state=None, splitter='best'),
                                 n_estimators=150, learning_rate=.5, loss='linear', random_state=None)
#clf = gaussian_process.GaussianProcess(corr='cubic', theta0=1e-2, thetaL=1e-4, thetaU=1e-1, random_start=100)
#clf = neighbors.KNeighborsRegressor(100, weights='uniform', algorithm = 'auto');clf_name='KNN_200'



#---Cross Validation---#
cv_preds = train.cross_validate(hstack([sparse.csr_matrix(dfTrn.urlid.values).transpose(),mtxTrn]),mtxTrnTarget.ravel(),folds=10,SEED=42,test_size=.1,clf=clf,clf_name=clf_name,pred_fg=True)  #may require mtxTrn.toarray()
train.cross_validate(mtxTrn,mtxTrnTarget.ravel(),folds=8,SEED=888,test_size=.1,clf=clf,clf_name=clf_name,pred_fg=False)  #may require mtxTrn.toarray()
train.cross_validate_temporal(mtxTrn,mtxTest,mtxTrnTarget.ravel(),mtxTestTarget.ravel(),clf=clf,clf_name=clf_name,pred_fg=False)
train.cross_validate_using_benchmark('global_mean',dfTrn, mtxTrn,mtxTrnTarget,folds=20)

#---Calculate the degree of variance between ground truth and the mean of the CV predictions.----#
#---Returns a list of all training records with their average variance---#
train.calc_cv_preds_var(dfTrn,cv_preds)

#--Use estimator for predictions--#
dfTest, clf = train.predict(mtxTrn,mtxTrnTarget.ravel(),mtxTest,dfTest,clf,clf_name) #may require mtxTest.toarray()
dfTest, clf = train.predict(mtxTrn.todense(),mtxTrnTarget.ravel(),mtxTest.todense(),dfTest,clf,clf_name) #may require mtxTest.toarray()

#--Save predictions to file--#
data_io.save_predictions(dfTest,clf_name,'_Views_Rich_GBM_GoogAPIcleaned_Source&Nbr&Weekend&Hrs&DescrLen',submission_no)
data_io.save_predictions(dfTest,clf_name,'_Votes_Rich_SGD_GoogAPIcleaned_Source&Nbr&Weekend&Hrs&DescrLen&LongLat&Tagtype',submission_no)
data_io.save_predictions(dfTest,clf_name,'_Cmts_Rich_GBM_GoogAPIcleaned_Source&Nbr&Weekend&Hrs&DescrLen&LongLat',submission_no)
data_io.save_predictions(dfTest,clf_name,'_Views_Oak_GBM_GoogAPIcleaned_Source&Nbr&Weekend&Hrs&DescrLen&Longlat&Tagtype',submission_no)
data_io.save_predictions(dfTest,clf_name,'_Votes_Oak_LassoCV10_GoogAPIcleaned_Source&Nbr&Weekend&Hrs&DescrLen&LongLat&Tagtype&TagtypeFG',submission_no)
data_io.save_predictions(dfTest,clf_name,'_Views_NH_GBM_GoogAPIcleaned_Source&Nbr&Weekend&Hrs&DescrLen&TotInc&LongLat',submission_no)
data_io.save_predictions(dfTest,clf_name,'_Views_Chi_LassoCV10_Source&Nbr&Weekend&Hrs&DescrLen&TagtypeFg',submission_no)

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
data_io.save_cached_object(clf,'rf_500_TextAll')

#--Load a model from joblib file--#
data_io.load_cached_object('Models/040513--rf_500_TextAll.joblib.pk1')

#--Save text feature names list for later reference--#
data_io.save_text_features("Data/text_url_features.txt",tfidf_vec.get_feature_names())

#--Save a dataframe to CSV--#
filename = 'Data/Test_Holidays'+'.csv'
###Fill missing values with 0 to preserve all columns for csv export
dfTest.ix[:,['holiday_fg','holiday_word','urlid']].to_csv(filename, index=True,  sep='\t',encoding='utf-8')