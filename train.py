'''
Functions for training classifiers, performing cross validation, and making predictions
'''
__author__ = 'Bryan Gregory'
__email__ = 'bryan.gregory1@gmail.com'
__date__ = '09-28-2013'

import ml_metrics
import time
from datetime import datetime
from sklearn import (metrics, cross_validation, linear_model, preprocessing)
from sklearn.externals import joblib
import numpy as np
from scipy import sparse

#---Traditional K-Fold Cross Validatio----#
def cross_validate(mtxTrn,mtxTarget,folds=5,SEED=42,test_size=.15,clf=None,clf_name=None,pred_fg=False):
    fold_scores = []
    SEED = SEED *  time.localtime().tm_sec
    start_time = datetime.now()
    print "CV started at:", datetime.now().strftime("%d-%m-%y %H:%M")

    #If predictions are wanted, intialize the dict so that its length will match all records in the training set,
    #even if not all records are predicted during the CV (randomness is a bitch)
    if pred_fg == True:
        cv_preds = {key[0]:[] for key in mtxTrn.getcol(0).toarray()}

    for i in range(folds):
        ##For each fold, create a test set (test_cv) by randomly holding out test_size% of the data as CV set
        train_cv, test_cv, y_target, y_true = \
           cross_validation.train_test_split(mtxTrn, mtxTarget, test_size=test_size, random_state=i*SEED+1)

        #Only for the 3-1-1 competition, refactor y_true to normal to accommodate for the log(label-1) labels trick
        y_true = [np.exp(x)-1 for x in y_true]

        #if predictions are wanted, parse off the first row from train and test cv sets. First row contains ID
        if pred_fg == True:
            #TODO: create dense matrix copies for the clf's that only use dense matrices
            train_cv = sparse.csr_matrix(train_cv)[:,1:]
            test_cv2 = sparse.csr_matrix(test_cv)[:,1:]
            test_cv = sparse.csr_matrix(test_cv)[:,1:]
        ##If you want to perform feature selection / hyperparameter optimization, this is where you want to do it

        ##If pre-calculated predictions have been passed use them, otherwise use the classifier to make predictions
        if clf is None:
            preds = test_cv
            print 'ERROR: NO CLASSIFIER!'
        else:
            clf.fit(train_cv, y_target)
            preds = clf.predict(test_cv)
            #Only for the 3-1-1 competition, refactor preds to normal to accommodate for the log(label-1) labels trick
            preds = [np.exp(x) - 1 for x in preds]
            #set any negative predictions to 0 as we cannot have negative votes, views, or comments
            preds = [0 if x < 0 else x for x in preds]
        ##For each fold, score the prediction by measuring the error using the chosen error metric
        score = ml_metrics.rmsle(y_true, preds)
        fold_scores += [score]
        print "RMLSE (fold %d/%d): %f" % (i + 1, folds, score)
        ##IF we want to record predictions, then for each fold add the predictions to the cv_preds dict for later output
        if pred_fg == True:
            for i in range(0,test_cv2.shape[0]):
                if test_cv2.getcol(0).toarray()[i][0] in cv_preds.keys():
                    cv_preds[test_cv2.getcol(0).toarray()[i][0]] += [preds[i]]
                else:
                    cv_preds[test_cv2.getcol(0).toarray()[i][0]] = [preds[i]]
    ##Now that folds are complete, calculate and print the results
    finish_time = datetime.now()
    print "========================================="
    print "Total mean: %f" % (np.mean(fold_scores))
    print "Total std dev: %f" % (np.std(fold_scores))
    print "Total max/min: %f/%f" % (np.max(fold_scores),np.min(fold_scores))
    print "========================================="
    print "%d fold CV completed at: %s.  Total runtime: %s" % ((folds), datetime.now().strftime("%H:%M"),str(finish_time-start_time))
    if pred_fg == True:
        return cv_preds

#---Temporal cross validation---#
def cross_validate_temporal(mtxTrn,mtxTest,mtxTrnTarget,mtxTestTarget,clf=None,clf_name=None,pred_fg=False):
    fold_scores = []
    start_time = datetime.now()
    print "CV started at:", datetime.now().strftime("%d-%m-%y %H:%M")
    print "========================================="

    #If predictions are wanted, intialize the dict so that its length will match all records in the training set,
    #even if not all records are predicted during the CV (randomness is a bitch)
    if pred_fg == True:
        cv_preds = {key[0]:[] for key in mtxTrn.getcol(0).toarray()}

    train_cv = mtxTrn
    test_cv = mtxTest
    y_target = mtxTrnTarget
    y_true = mtxTestTarget

    #Add pre processing rules here
    ##Only for the 3-1-1 competition, refactor y_true to normal to accommodate for the log(label-1) labels trick
    y_true = [np.exp(x)-1 for x in y_true]
    ##scalar
    #y_true = [x*.48 for x in y_true]
    #y_true = [x*.08 for x in y_true]
    #y_true = [x*.8 for x in y_true]

    #if predictions are wanted, parse off the first row from train and test cv sets. First row contains ID
    if pred_fg == True:
        #TODO: create dense matrix copies for the clf's that only use dense matrices
        train_cv = sparse.csr_matrix(train_cv)[:,1:]
        test_cv2 = sparse.csr_matrix(test_cv)[:,1:]
        test_cv = sparse.csr_matrix(test_cv)[:,1:]
    ##If you want to perform feature selection / hyperparameter optimization, this is where you want to do it

    ##If pre-calculated predictions have been passed use them, otherwise use the classifier to make predictions
    if clf is None:
        preds = test_cv
        print 'ERROR: NO CLASSIFIER!'
    else:
        clf.fit(train_cv, y_target)
        preds = clf.predict(test_cv)
        #---Add post processing rules here---#
        ##Only for the 3-1-1 competition, refactor y_true to normal to accommodate for the log(label-1) labels trick
        preds = [np.exp(x)-1 for x in preds]
        ##apply scalar
        preds = [x*.7 for x in preds]  #Views
        #preds = [x*.999 for x in preds]  #Votes
        #preds = [x*.9 for x in preds]  #Cmts
        ##set any negative predictions to 0 as we cannot have negative views, or comments
        preds = [0 if x < 0 else x for x in preds]
        ##set <1 predictions to 1 as we cannot have < 1 votes
        #preds = [1 if x < 1 else x for x in preds]
    ##score the prediction by measuring the error using the chosen error metric
    score = ml_metrics.rmsle(y_true, preds)
    print "Error Measure:" , score
    ##IF we want to record predictions, then for each fold add the predictions to the cv_preds dict for later output
    if pred_fg == True:
        for i in range(0,test_cv2.shape[0]):
            if test_cv2.getcol(0).toarray()[i][0] in cv_preds.keys():
                cv_preds[test_cv2.getcol(0).toarray()[i][0]] += [preds[i]]
            else:
                cv_preds[test_cv2.getcol(0).toarray()[i][0]] = [preds[i]]
    finish_time = datetime.now()
    print "========================================="
    print "CV completed at: %s.  Total runtime: %s" % (datetime.now().strftime("%H:%M"),str(finish_time-start_time))
    if pred_fg == True:
        return cv_preds

def cross_validate_using_benchmark(benchmark_name, dfTrn, mtxTrn,mtxTarget,folds=5,SEED=42,test_size=.15):
    fold_scores = []
    SEED = SEED *  time.localtime().tm_sec
    start_time = datetime.now()
    print "CV w/ avgs started at:", datetime.now().strftime("%d-%m-%y %H:%M")
    print "========================================="
    #scores = cross_validation.cross_val_score(clf, mtxTrn, mtxTarget, cv=folds, random_state=i*SEED+1, test_size=test_size, scoring)
    for i in range(folds):
        #For each fold, create a test set (test_holdout) by randomly holding out X% of the data as CV set, where X is test_size (default .15)
        train_cv, test_cv, y_target, y_true = cross_validation.train_test_split(mtxTrn, mtxTarget, test_size=test_size, random_state=SEED*i+10)

        #Calc benchmarks and use them to make a prediction
        benchmark_preds = 0
        if benchmark_name =='global_mean':
            benchmark_preds = [13.899 for x in test_cv]
        if benchmark_name =='all_ones':
            #find user avg stars mean
            benchmark_preds = [1 for x in test_cv]
        if benchmark_name =='9999':
            #find user avg stars mean
            benchmark_preds = [9999 for x in test_cv]
        print 'Using benchmark %s:' % (benchmark_name)

        #For this CV fold, measure the error (distance between the predictions and the actual targets)
        score = ml_metrics.rmsle(y_true, benchmark_preds)
        #print score
        fold_scores += [score]
        print "RMSLE (fold %d/%d): %f" % (i + 1, folds, score)

    ##Now that folds are complete, calculate and print the results
    finish_time = datetime.now()
    print "========================================="
    print "Total mean: %f" % (np.mean(fold_scores))
    print "Total std dev: %f" % (np.std(fold_scores))
    print "Total max/min: %f/%f" % (np.max(fold_scores),np.min(fold_scores))
    print "========================================="
    print "%d fold CV completed at: %s.  Total runtime: %s" % ((folds), datetime.now().strftime("%H:%M"),str(finish_time-start_time))

def predict(mtxTrn,mtxTarget,mtxTest,dfTest,clf,clfname):
    start_time = datetime.now()

    #fit the classifier
    clf.fit(mtxTrn, mtxTarget)

    #make predictions on test data and store them in the test data frame
    #if clfname == 'log':
    dfTest['predictions_'+clfname] = [x for x in clf.predict(mtxTest)]
    #else:
    #dfTest['predictions_'+clfname] = [x for x in clf.predict(mtxTest)]

    ##This section only for the 3-1-1 competition, to refactor predictions back to normal
    dfTest['predictions_'+clfname] = [np.exp(x) - 1 for x in dfTest['predictions_'+clfname]]

    #make all preds >0
    dfTest['predictions_'+clfname] = [0 if x < 0 else x for x in dfTest['predictions_'+clfname]]

    #if predicting votes, make all preds >1
    #dfTest['predictions_'+clfname] = [1 if x < 1 else x for x in dfTest['predictions_'+clfname]]

    #print "Coefs for",clfname,clf.coef_
    finish_time = datetime.now()
    print "Predictions made.  Mean:",dfTest['predictions_'+clfname].mean()
    print "Completed at %s. Total runtime: %s" % (datetime.now().strftime("%H:%M"),str(finish_time-start_time))
    return dfTest,clf

#---Calculate the degree of variance between ground truth and the mean of the CV predictions.----#
#---Adds the average cv variance to the training dataframe for later analysis--------------------#
def calc_cv_preds_var(df, cv_preds):
    df['cv_preds_var'] = ''
    df['cv_preds_mean'] = ''
    for key in cv_preds.keys():
        df['cv_preds_var'][df.urlid == key] = abs(df[df.urlid == key].label.values[0] - np.mean(cv_preds[key]))
        df['cv_preds_mean'][df.urlid == key] = np.mean(cv_preds[key])




