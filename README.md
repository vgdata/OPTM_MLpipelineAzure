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
(iv)Hyperparameters were chosen with Random Sampling and regression models were used for traning with hyperparameters (C, max_iter)
(v) once the experiment is completed, BanditPolicy was used for early termination. This was we can save further use of resources by
stopping the hyperparameter run. (vi)  using hyperparametrs, the best model was observed and then saved.

Pipeline was ran several times, with Hyperdrive configuration to improve our Accuracy of the model and once
satisfied we registerd our model for future use. In this case the best model was generated using this hyperparameters
**(C = (0, 1), max_iter = randint(100')** and give us an  **Accuracy of 0.911**

we chose **C** and **max_iter** parameters with random sampling **RandomParameterSampling** 
to try different possible configuration with discrete 'C' and 'max_iter' values

**early stopping policy **
We then define our termination Policy for every run using **BanditPolicy** based on a slack factor  of 0.1 equal to 
This helps to reduce the number of poorly performing runs and hence the cost.

## AutoML

After cleaning the data by one-hot encoding, traninging ans testing set were defined. In the AutoML configuration, the primary metric was defined as 
"Accuracy". Then experiment was submitted with submit method and Best Model was found, which was registered for future use.  
the best model was generated using **VotingEnsemble Algorithm** which involves summing the predictions made by multiple other
 classification models and give us an  **Accuracy of 0.9170**

## Pipeline comparison
The AutoML  approach generated multiple models and tested/optimized on different hyperparameters.
A better accuracy is obtained (0.917 or 91.7%), which is a primary metric in the given task. The execution time 
was longer for AutoML , which was approximatly 40 minutes, since it ran more models.

In Logistic regression model, the accuracy was 0.911 and it was developed with hyperdrive.
Here the execution time was  6 minutes, but accuracy was less.


## Future work
More algorithms can be added to Scikit-learn process to test other configuration  and tune hyperparameters.
Some preprocessing techniques like feature selection can be used. We can have different primary metrics and do comparion
to get overall good results.

## Proof of cluster clean up

At the end of the pipeline run, cluster was cleaned.