Prize Winning Model for Kaggle See-Click-Fix Contest
============================================================
Prize-winning solution for Kaggle contest "SeeClickPredictFix". The purpose of the contest was to train a model (as scored by RMSLE) using supervised learning that will accurately predict the views, votes, and comments that an issue posted to the www.seeclickfix.com website will receive. This code developed by me uses a segment based ensemble to generate predictions for the 3 targets (views, votes, and comments). My teammate and I used this model in combination with his to create a winning model for the contest, defeating >500 other teams and winning a purse of $1,0000.

More contest info: http://www.kaggle.com/c/see-click-predict-fix

In-depth code description here:  http://bryangregory.com/Kaggle/DocumentationforSeeClickFix.pdf

"How I Did It" blog post here: http://bryangregory.com/Kaggle/Kaggle-SeeClickFix-HowIDidIt.pdf

Ensemble model code used for combining our models: https://github.com/theusual/kaggle-seeclickfix-ensemble

My teammate's individual model: https://github.com/beegieb/kaggle_see_click_fix

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





