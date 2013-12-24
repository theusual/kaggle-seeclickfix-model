"""
Classes and functions for working with base models and ensembles.
"""
__author__ = 'bgregory'
__email__ = 'bryan.gregory1@gmail.com'
__date__ = '11-23-2013'

import ml_metrics
import data_io
import numpy as np
import pandas as pd
from sklearn import (metrics, cross_validation, linear_model, ensemble, tree, preprocessing, svm, neighbors, gaussian_process, naive_bayes, neural_network, pipeline, lda)


########################################################################################
class Model(object):
    """Base class for all models:  stand-alone independent models, base models,
    and ensemble models.

    Parameters
    ----------
    model_name: string, required
        Descriptive name of model for use in logging and output

    estimator_class: string, required
        SKLearn estimator class for model fitting and training

    features: dictionary, required
        Features of the model.  Key is feature name, value is a dictionary with
        'train' and 'test' arrays.
            Ex.- model.features['foo_feature']['train'] will return an
            array with the values in training set for foo_feature

    target: string, optional (default = global_target from settings file)
        Target variable (column name) for this model

    segment: string, optional (default = none)
        Segment of data for this model to use

    estimator_params: dictionary, optional (default=none, which passes to SKLearn defaults for that estimator)
        Parameters of the estimator class

    postprocess_scalar: float, optional (default=0)
        Scalar to apply to all predictions after model predicting, useful for calibrating predictions

    Attributes
    ----------
    """

    def __init__(self, model_name, target, segment, estimator_class, estimator_params, features, postprocess_scalar):
        self.model_name = model_name
        self.target = target
        self.segment = segment
        self.estimator_class = estimator_class
        self.set_estimator(estimator_class, estimator_params)
        self.set_features(features)
        self.postprocess_scalar = round(np.float32(postprocess_scalar), 4)
    def set_estimator(self, estimator_class, estimator_params):
            self.estimator = eval(estimator_class)()
            for param in estimator_params:
                #Convert any numerical parameters from the required JSON string type
                try:
                    float(param)
                except:
                    pass
                setattr(self.estimator, param, estimator_params[param])
    def set_features(self, features):
        """Initialize dictionary of features where keys are the feature names and values are an empty
        list for storing the training and testing array/matrix"""
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

########################################################################################
class EnsembleAvg (object):
    """Loads already calculated predictions from individual models in the form of CSV files, then applies
    average weights to each individual model to create an ensemble model.
    If predictions are for a cross-validation, then true target values can be loaded and the ensemble can be scored
    using given weights or using optimally derived weights.

    Attributes:
    df_models = List containing each individual model's predictions
    id = unique ID for each record
    targets = List containing the target (or targets) for the predictions
    df_true = Pandas DataFrame containing the true values for the predictions, only required if performing CV
    """
    def __init__(self, targets, id):
        self.sub_models = []
        self.sub_models_segment = []
        self.targets = targets
        self.id = id
    def load_models_csv(self,filepath, model_no = None):
        """
        Load predictions from an individual sub model into a dataframe stored in the sub_models list, if no model_no is given
        then load data into next available index.  Valid source is a CSV file.
        """
        try:
            if model_no == None:
                model_no = len(self.sub_models)
                self.sub_models.append(data_io.load_flatfile_to_df(filepath, delimiter=''))
            else:
                self.sub_models[model_no]=data_io.load_flatfile_to_df(filepath, delimiter='')
            print "Model loaded into index %s" % str(model_no)
        except IndexError:
            raise Exception("Model number does not exist. Model number given, %s, is out of index range." % str(model_no))

    def load_true_df(self,df):
        """
        Load true target values (ground truth) into a dataframe attribute from an in-memory dataframe object.
        """
        if type(df) != pd.core.frame.DataFrame:
            raise Exception("Object passed, %s, is not a Dataframe. Object passed is of type %s" % (df, type(df)))
        elif self.id not in df.columns:
            raise Exception("Dataframe passed, %s, does not contain unique ID field: %s" % (df, self.id))
        elif not all(x in df.columns for x in self.targets):
            raise Exception("Dataframe passed, %s, does not contain all target variables: %s" % (df, self.targets))
        else:
            self.df_true = df.copy()
            print "True value for target variables successfully loaded into self.df_true"
    def load_df_true_segment(self,df):
        """
        For segmented data.
        Load true target values (ground truth) into a dataframe attribute from an in-memory dataframe object.
        """
        if type(df) != pd.core.frame.DataFrame:
            raise Exception("Object passed, %s, is not a Dataframe. Object passed is of type %s" % (df, type(df)))
        elif self.id not in df.columns:
            raise Exception("Dataframe passed, %s, does not contain unique ID field: %s" % (df, self.id))
        elif not all(x in df.columns for x in self.targets):
            raise Exception("Dataframe passed, %s, does not contain all target variables: %s" % (df, self.targets))
        else:
            self.df_true_segment = df.copy()
            print "True value for target variables successfully loaded into self.df_true_segment"
    def sort_dataframes(self,sortcolumn):
        """
        Sort all data frame attributes of class by a given column for ease of comparison.
        """
        try:
            for i in range(len(self.sub_models)):
                self.sub_models[i] = self.sub_models[i].sort(sortcolumn)
            if "df_true" in dir(self):
                self.df_true = self.df_true.sort(sortcolumn)
            if "df_true_segment" in dir(self):
                self.df_true_segment = self.df_true_segment.sort(sortcolumn)
        except KeyError:
            raise Exception("Sort failed.  Column %s not found in all dataframes." % (sortcolumn))
    def transform_targets_log(self):
        """
        Apply natural log transformation to all targets (both predictions and true values)
        """
        for target in self.targets:
            if "df_true" in dir(self):
                self.df_true[target] = np.log(self.df_true[target] + 1)
            if "df_true_segment" in dir(self):
                self.df_true_segment[target] = np.log(self.df_true_segment[target] + 1)
            for i in range(len(self.sub_models)):
                self.sub_models[i][target] = np.log(self.sub_models[i][target] + 1)
            for i in range(len(self.sub_models_segment)):
                self.sub_models_segment[i][target] = np.log(self.sub_models_segment[i][target] + 1)
    def transform_targets_exp(self):
        """
        Apply exp transformation (inverse of natural log transformation) to all targets (both predictions and true values)
        """
        for target in self.targets:
            if "df_true" in dir(self):
                self.df_true[target] = np.exp(self.df_true[target])-1
            if "df_true_segment" in dir(self):
                self.df_true_segment[target] = np.exp(self.df_true_segment[target])-1
            if 'df_ensemble' in dir(self):
                self.df_ensemble[target] = np.exp(self.df_ensemble[target])-1
            if 'df_ensemble_segment' in dir(self):
                self.df_ensemble_segment[target] = np.exp(self.df_ensemble_segment[target])-1
            for i in range(len(self.sub_models)):
                self.sub_models[i][target] = np.exp(self.sub_models[i][target])-1
            for i in range(len(self.sub_models_segment)):
                self.sub_models_segment[i][target] = np.exp(self.sub_models_segment[i][target]) -1
    def score_rmsle(self,df,df_true):
        """
        Calculate CV score of predictions in given dataframe using RMSLE metric.  Score individually for each target and
        total for targets.  Must have df_true loaded prior to running.
        """
        all_true = []
        all_preds = []
        target_scores = []
        #Transform predictions back to normal space for scoring
        self.transform_targets_exp()
        for target in self.targets:
            all_true.append(df_true[target].tolist())
            all_preds.append(df[target].tolist())
            target_score = ml_metrics.rmsle(df_true[target], df[target])
            target_scores.append(target_score)
            print "RMSLE score for %s: %f" % (target,target_score)
        print "Total RMSLE score: %f" % (ml_metrics.rmsle(all_true, all_preds))
        #Transform predictions to log space again for averaging
        self.transform_targets_log()
    def create_ensemble(self,sub_model_indexes, weights):
        """
        Create ensemble from the given sub models using average weights.
        Sub_model_indexes is a list of indexes to use for the sub_models list.
        Weights is a list of dictionaries with given averages for each target, its ordering must correspond to
        the order of sub_model_indexes.
        Ex. -   >>>  weights = [{"target1":.5,"target2":.5},{"target1":.25,"target2":.75}]
        """
        if len(sub_model_indexes) != len(weights):
            raise Exception("Ensemble failed. Number of sub models, %d, is not equal to number of weights, %d.") \
                            % (len(sub_model_indexes), len(weights))
        else:
            #Create new data frame ensemble
            self.df_ensemble = self.sub_models[0].copy()
            for target in self.targets:
                self.df_ensemble[target] = 0
                for submodel in sub_model_indexes:
                    for idx in self.df_ensemble.index:
                        self.df_ensemble[target][idx] += self.sub_models[submodel][target] * weights[submodel][target]
    def create_ensemble_segment(self,sub_model_indexes, weights):
        """
        Create ensemble for a certain segment, from the given sub models using average weights.
        Sub_model_indexes is a list of indexes to use for the sub_models list.
        Weights is a list of dictionaries with given averages for each target, its ordering must correspond to
        the order of sub_model_indexes.
        Ex. -   >>>  weights = [{"target1":.5,"target2":.5},{"target1":.25,"target2":.75}]
        """
        if len(sub_model_indexes) != len(weights):
            raise Exception("Ensemble failed. Number of sub models, %d, is not equal to number of weights, %d.") \
                            % (len(sub_model_indexes), len(weights))
        else:
            #Create new data frame ensemble
            self.df_ensemble_segment = self.sub_models_segment[0].copy()
            for target in self.targets:
                self.df_ensemble_segment[target] = 0
                for submodel in sub_model_indexes:
                    self.df_ensemble_segment[target] += self.sub_models_segment[submodel][target] * weights[submodel][target]
    def calc_weights(self,sub_model_indexes, step_size):
        """
        Calculate optimal weights to use in averaged ensemble using the given sub-models and given score metric
        """
        for target in self.targets:
            while diff < 0:
                score

"""

            if source == "df":
                if type(df) == pd.core.frame.DataFrame:
                    if model_no == None:
                        model_no = len(self.df_models)
                        self.df_models.append(df)
                        print source,3
                    else:
                        self.df_models[model_no]=df
                else:
                    raise Exception("Source passed is not a valid dataframe object: %s." % df)
            else:
                raise Exception("Incorrect source passed: %s." % source)
"""