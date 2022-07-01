# Credit Risk Analysis

## Overview

In this project, we were given a large dataset of loan applications with all collected info included. The task is to take this extremely imbalanced
dataset (there are far more low risk applications than there are high risk ones), and use machine learning to predict whether an application 
should be labeled as low or high risk. 

### Resampling Techniques

We used 4 different resampling techniques to first make the dataset more balanced in the credit_risk_resampling file:

* Oversampling with RandomOverSampler
* SMOTE Oversampling
* Undersampling using the CLusterCentroids method
* Combination of oversampling and undersampling using SMOTEENN

After each resampling, we used a logistic regression model to make our predictions, and assessed the performance of each model.

### Ensemble Learning Techniques

In the credit_risk_ensemble file, we used two different ensemble learning techniques:

* The BalancedRandomForestClassifier
* The EasyEnsembleClassifier

Each model was assessed using the same process as in the resampling file.

## Results

### Oversampling with RandomOverSampler

Using the RandomOverSampler method randomly selects instances of the minority class and ads it to the training set until the classes are balanced.
 

