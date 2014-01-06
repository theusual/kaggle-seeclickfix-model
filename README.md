Prize Winning Model for Kaggle See-Click-Fix Contest
============================================================
Prize winning solution to the SeeClickFix contest hosted on Kaggle.  Developed by teammates Bryan Gregory and Miroslaw Horbal.

More contest info here: http://www.kaggle.com/c/see-click-predict-fix
In-depth code description here:  http://bryangregory.com/Kaggle/DocumentationforSeeClickFix.pdf
"How I Did It" blog post here: http://bryangregory.com/Kaggle/Kaggle-SeeClickFix-HowIDidIt.pdf

Description
=============
Code for generating an ensemble submission using base submission files generated by our team's individual models, which are located in subfolders under /Bryan/ and /Miroslaw/. The code uses segment based averaging to combine the submissions, based on 5 primary segments identified in the data (remote_api, Chicago, Oakland, New Haven, and Richmond). 

The code also has options to perform averaging in log space (log transformations are performed prior to averaging) and to apply final scalars to the ensemble predictions. Both of these settings were found to increase model accuracy and were used in our winning submission, therefore those settings are defaulted to on.

Independent code repositories for my teammate's model and the ensemble code used for combining our models are found here:

Ensemble model code: https://github.com/theusual/kaggle-seeclickfix-ensemble

Miroslaw's model: https://github.com/beegieb/kaggle_see_click_fix

Usage
========
SETTINGS.json contains filepaths along with other misc. settings. 
MODELS.json contains many configurable options for each segment's base model including each base model's estimator, estimator parameters, features, and scalars. 
Both SETTINGS.json and MODELS.json have been pre-configured to use the settings of the winning model submission, but settings can be changed as needed prior to running.

To run:

      >>> python main.py
      
A submission file will be generated that defaults to being created in "Output/bryan_test_predictions.csv".  This can be changed in SETTINGS.json.

Requirements
================
      PYTHON >= 2.6
      PANDAS >= .11
      NUMPY
      SKLEARN >= .13
      SCIPY >= .12




Prize-winning model code for Kaggle contest "SeeClickPredictFix". The purpose of the contest is to train a model (as scored by RMSLE) using supervised learning that will accurately predict the views, votes, and comments that an issue posted to the www.seeclickfix.com website will receive.  This code uses a segment based ensemble to generate predictions for the 3 targets (views, votes, and comments).  My teammate and I used this model in combination with his to create a winning model for the contest, defeating >500 other teams.
