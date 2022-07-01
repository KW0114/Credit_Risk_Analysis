# Credit Risk Analysis

## Overview

In this project, we were given a large dataset of loan applications with all collected info included. The task is to take this extremely imbalanced
dataset (there are far more low risk applications than there are high risk ones), and use machine learning to predict whether an application 
should be labeled as low or high risk. 

### Resampling Techniques

We used 4 different resampling techniques to first make the dataset more balanced in the credit_risk_resampling file:

* Oversampling with RandomOverSampler
* SMOTE Oversampling
* Undersampling using the ClusterCentroids method
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
 
The accuracy score of the logictic regression model using this resampling method was 64.7%:

![accuracy score](https://github.com/KW0114/Credit_Risk_Analysis/blob/52a56a3672c09033b6b434e993ba109ed07080df/Screenshots/Random_Oversampling_Accuracy_Score.png)

Refer to the following classification report for the precision and recall scores for each classification:

![classification report](https://github.com/KW0114/Credit_Risk_Analysis/blob/3f0c348af68a3f63346a045747b7de596da6bfb5/Screenshots/Random_Oversampling_Report.png)

For the high risk class:

* The precision is 1%
* The recall is 57%

For the low risk class:

* The precision is 100%
* The recall is 65%

### SMOTE Oversampling

Using the synthetic minority oversampling technique (SMOTE), new instanced are interpolated to add to the minority class.

The accuracy score of the logictic regression model using this resampling method was 65%:

![accuracy score] (https://github.com/KW0114/Credit_Risk_Analysis/blob/3f0c348af68a3f63346a045747b7de596da6bfb5/Screenshots/SMOTE_Accuracy_Score.png)

Refer to the following classification report for the precision and recall scores for each classification:

![classification report](https://github.com/KW0114/Credit_Risk_Analysis/blob/3f0c348af68a3f63346a045747b7de596da6bfb5/Screenshots/SMOTE_Report.png)

For the high risk class:

* The precision is 1%
* The recall is 62%

For the low risk class:

* The precision is 100%
* The recall is 67%

### Undersampling Using the ClusterCentroids Method

Cluster centroid undersampling identifies clusters of the majority class, then generates synthetic data points (centroids), that are representative of the clusters.

The accuracy score of the logictic regression model using this resampling method was 54%:

![accuracy score] (https://github.com/KW0114/Credit_Risk_Analysis/blob/3f0c348af68a3f63346a045747b7de596da6bfb5/Screenshots/ClusterCentroids_Accuracy_Score.png)

Refer to the following classification report for the precision and recall scores for each classification:

![classification report](https://github.com/KW0114/Credit_Risk_Analysis/blob/3f0c348af68a3f63346a045747b7de596da6bfb5/Screenshots/ClusterCentroids_Report.png)

For the high risk class:

* The precision is 1%
* The recall is 66%

For the low risk class:

* The precision is 100%
* The recall is 42%

### Combination of Oversampling and Undersampling Using SMOTEENN

SMOTEENN is a two step process:
1. First, use the SMOTE method to oversample the data
2. Then, clean the resulting data. If the two nearest neighbors of a data point beling to different classes, drop that data point.

The accuracy score of the logictic regression model using this resampling method was 62%:

![accuracy score] (https://github.com/KW0114/Credit_Risk_Analysis/blob/3f0c348af68a3f63346a045747b7de596da6bfb5/Screenshots/SMOTEENN_Accuracy_Score.png)

Refer to the following classification report for the precision and recall scores for each classification:

![classification report](https://github.com/KW0114/Credit_Risk_Analysis/blob/3f0c348af68a3f63346a045747b7de596da6bfb5/Screenshots/SMOTEENN_Report.png)

For the high risk class:

* The precision is 1%
* The recall is 69%

For the low risk class:

* The precision is 100%
* The recall is 54%

### The BalancedRandomForestClassifier

The accuracy score of this random forest classifier is 79%:

![accuracy score] (https://github.com/KW0114/Credit_Risk_Analysis/blob/3f0c348af68a3f63346a045747b7de596da6bfb5/Screenshots/BRF_Accuracy_Score.png)

Refer to the following classification report for the precision and recall scores for each classification:

![classification report](https://github.com/KW0114/Credit_Risk_Analysis/blob/3f0c348af68a3f63346a045747b7de596da6bfb5/Screenshots/BRF_Report.png)

For the high risk class:

* The precision is 3%
* The recall is 71%

For the low risk class:

* The precision is 100%
* The recall is 88%

### The EasyEnsembleClassifier

The accuracy score of the easy ensemble classifier is 91%:

![accuracy score] (https://github.com/KW0114/Credit_Risk_Analysis/blob/3f0c348af68a3f63346a045747b7de596da6bfb5/Screenshots/EEC_Accuracy_Score.png)

Refer to the following classification report for the precision and recall scores for each classification:

![classification report](https://github.com/KW0114/Credit_Risk_Analysis/blob/3f0c348af68a3f63346a045747b7de596da6bfb5/Screenshots/BRF_Report.png)

For the high risk class:

* The precision is 7%
* The recall is 89%

For the low risk class:

* The precision is 100%
* The recall is 94%

## Summary

First, we used 4 different resampling methods before using those resampled training sets in our logistic model. There is a negligible difference
between each of these resampling methods. The precision and recall scores are very similar for each category between each method. The accuracy scores
are also all within about 11% of each other. 

However, there is a noticeable difference when we use ensamble learning instead of logistic regression.
As seen above, the accuracy scores were 79% and 91%, a high better score than any of our logistic regression models. 
The precision of these models in the high risk class was also slightly higher than 1%, that ouf our logistic regression models. 

If I were to recommend a model, it would be the EasyEnsembleClassifier. This is because it not only has a 91% accuracy score, but the recall for each 
class is also pretty good (89% for high risk, and 94% for low risk). This model is 89% likely to correctly classify a high risk application, and 94%
likely to correctly classify a low risk application. 







