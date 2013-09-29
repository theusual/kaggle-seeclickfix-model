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

def cross_validate(mtxTrn,mtxTarget,folds=5,SEED=42,test_size=.15,clf=None,clf_name=None,pred_fg=False):
    fold_scores = []
    SEED = SEED *  time.localtime().tm_sec
    start_time = datetime.now()
    print "CV started at:", datetime.now().strftime("%d-%m-%y %H:%M")
    print "========================================="

    #If predictions are wanted, intialize the dict so that its length will match all records in the training set,
    #even if not all records are predicted during the CV (randomness is a bitch)
    if pred_fg == True:
        cv_preds = {key[0]:[] for key in mtxTrn.getcol(0).toarray()}

    for i in range(folds):
        ##For each fold, create a test set (test_cv) by randomly holding out X% of the data as CV set, where X is test_size (default .15)
        train_cv, test_cv, y_target, y_true = cross_validation.train_test_split(mtxTrn, mtxTarget, test_size=test_size, random_state=i*SEED+1)

        train_cv = sparse.csr_matrix(train_cv)[:,1:]
        test_cv2 = sparse.csr_matrix(test_cv)[:,1:]
        ##If you want to perform feature selection / hyperparameter optimization, this is where you want to do it

        ##If pre-calculated predictions have been passed use them, otherwise use the classifier to make predictions
        if clf is None:
            preds = test_cv
            print 'ERROR: NO CLASSIFIER!'
        else:
            clf.fit(train_cv, y_target)
            preds = clf.predict(test_cv2)
            preds = [0 if x < 0 else x for x in preds]
        ##For each fold, score the prediction by measuring the error using the chosen error metric
        score = ml_metrics.rmsle(y_true, preds)
        fold_scores += [score]
        print "RMLSE (fold %d/%d): %f" % (i + 1, folds, score)
        ##IF we want to record predictions, then for each fold add the predictions to the cv_preds dict for later output
        if pred_fg == True:
            for i in range(0,test_cv.shape[0]):
                if test_cv.getcol(0).toarray()[i][0] in cv_preds.keys():
                    cv_preds[test_cv.getcol(0).toarray()[i][0]] += [preds[i]]
                else:
                    cv_preds[test_cv.getcol(0).toarray()[i][0]] = [preds[i]]
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
    #fit the classifier
    clf.fit(mtxTrn, mtxTarget)

    #make predictions on test data and store them in the test data frame
    #if clfname == 'log':
    dfTest['predictions_'+clfname] = [x for x in clf.predict(mtxTest)]
    #else:
    #dfTest['predictions_'+clfname] = [x for x in clf.predict(mtxTest)]

    #print "Coefs for",clfname,clf.coef_
    print "Predictions made.  Mean:",dfTest['predictions_'+clfname].mean()
    return dfTest,clf

#---Calculate the degree of variance between ground truth and the mean of the CV predictions.----#
#---Adds the average cv variance to the training dataframe for later analysis--------------------#
def calc_cv_preds_var(df, cv_preds):
    df['cv_preds_var'] = ''
    df['cv_preds_mean'] = ''
    for key in cv_preds.keys():
        df['cv_preds_var'][df.urlid == key] = abs(df[df.urlid == key].label.values[0] - np.mean(cv_preds[key]))
        df['cv_preds_mean'][df.urlid == key] = np.mean(cv_preds[key])

