# Optimizing an ML Pipeline in Azure  

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
Models are created with PythonSDK and AutoML and then compared. The prediction will be about clients who will
subscrible a fixed term deposit

## Summary
**The given dataset contains marketing data of a bank and the sample dataset is provided by Azure at:
https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv

Best Performing Model: In this given dataset, the best performing model was VotingEnsemble (using Auto ML) with the accuracy
of 0.9170, which is better than the the logistic regression model (using hyperdrive) with 0.911

## Scikit-learn Pipeline
(i) Data is loaded as a very first step, which is imported from TabularDatasetFactory. (ii) Data cleaning: Hot encoding 
was used as columns have categorical data.  (iii)Data split was used as 0.7:0.3 for traning and testing datasets respectively
(iv)Hyperparameters were chosen with Random Sampling and regression models were used for training with hyperparameters (C, max_iter)
(v) once the experiment is completed, BanditPolicy was used for early termination. This was we can save further use of resources by
stopping the hyperparameter run. (vi)  using hyperparametrs, the best model was observed and then saved.

Pipeline was ran several times, with Hyperdrive configuration to improve our Accuracy of the model and once
satisfied we registerd our model for future use. In this case the best model was generated using this hyperparameters
**(C = (0, 1), max_iter = randint(100')** and give us an  **Accuracy of 0.911**

Parameters: 
Logistic Regression Model was used for training with hyperparameter tuning such as C and max_iter using HyperDrive.
A model parameter is a configuration variable that is internal to the model. They are required by the model when
 making predictions.Parameter C is a continuous or discrete parameter or  and max_iter parameter is Maximum number of iterations
taken to converge to  take full advantage of randomization.

 **C** and **max_iter** parameters with random sampling **RandomParameterSampling** 
 try different possible configuration with  'C' and 'max_iter' values, that tends
to maximize primary metric in defined search space.

Benefits of Random Parameter Sampling: 
Random Sampling is used to choose the hyperparameters as it is good for getting some values of hyperparameters
that one cannot guess intuitively. Random sampling would mean that we will cover most of the sample space and will get the best model.

In random parameter sampling, hyperparameter values are randomly selected from the
defined search space. Random sampling allows the search space to include both discrete and continuous hyperparameters.
A sample mean tends to be a good estimate of th a large dataset when samples are randomly selected over and over again
and calculates sample mean each time, so it gives the corerct values. 

**Benefits of the chosen early stopping policy **
We then define our termination Policy for every run using **BanditPolicy** based on a slack factor  of 0.1.
Bandit is based on slack factor/slack amount and evaluation interval. Bandit terminates runs where the primary metric
is not within the specified slack factor/slack amount (in this case  = 0.1) compared to the best performing run.
The early termination policy is applied at every interval when metrics are reported, starting at
evaluation interval of 2. Any run whose best metric is less than (1/(1+0.1) or 91% of the best performing run will be
terminated. This means poorly performing will be terminated, so overall cost will be reduced.


## AutoML

After cleaning the data by one-hot encoding, training ans testing set were defined. In the AutoML configuration, the primary metric was defined as 
"Accuracy". Then experiment was submitted with submit method and Best Model was found, which was registered for future use.  
the best model was generated using **VotingEnsemble Algorithm** which involves summing the predictions made by multiple other
classification models and give us an  **Accuracy of 0.9170**

Voting Ensemble Algorithm: A voting ensemble is an ensemble machine learning model, which combines the predictions from
multiple other models. It can be used for classification (Predictions are the majority vote of contributing models.)
or regression (Predictions are the average of contributing models). A Voting Ensemble is appropriate when multiple
models perform well on the task and agree with the results. This algorithm treats all models same which means all models
contribute equally to prediction. But some models may perform good in some situations or poor in other situation.
To address this issue, weighted average or weighted Voting is used.

## Pipeline comparison
The AutoML  approach generated multiple models and tested/optimized on different hyperparameters.
A better accuracy is obtained (0.917 or 91.7%), which is a primary metric in the given task. The execution time 
was longer for AutoML , which was approximatly 40 minutes, since it ran more models.

In Logistic regression model, the accuracy was 0.911 and it was developed with hyperdrive.
Here the execution time was  6 minutes, but accuracy was less.

## Proof of cluster clean up
I used used delete method of the compute object to remove the cluster. This helps to save cost on compute reesources.
<img src ="Screenshots/Compute_delete.png" alt = "compute_delete>
