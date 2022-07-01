# Credit Risk Analysis

## Overview

In this project, we were given a large dataset of loan applications with all collected info included. The task is to take this extremely imbalanced
dataset (there are far more low risk applications than there are high risk ones), and use machine learning to predict whether an application 
should be labeled as low or high risk. 

We used 4 different resampling techniques to first make the dataset more balanced:

* Oversampling with RandomOverSampler
* SMOTE Oversampling
* Undersampling using the CLusterCentroids method
* Combination of oversampling and undersampling using SMOTEENN