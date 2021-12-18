# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 16:19:38 2021

@author: Manajit Das
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import math
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

#prepare the train and test dataset
df=pd.read_csv('ExpDataMatrix.csv')
X=df.iloc[:, 1:-1]
y=df.iloc[:, -1]
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=42)

#customized class for removal of features with high correlation constant i.e. >0.8
class DropCollinear(BaseEstimator, TransformerMixin):
    def __init__(self, thresh):
        self.uncorr_columns = None
        self.thresh = thresh

    def fit(self, X, y):
        cols_to_drop = []

        # Find variables to remove
        X_corr = X.corr()
        large_corrs = X_corr>self.thresh
        indices = np.argwhere(large_corrs.values)
        indices_nodiag = np.array([[m,n] for [m,n] in indices if m!=n])

        if indices_nodiag.size>0:
            indices_nodiag_lowfirst = np.sort(indices_nodiag, axis=1)
            correlated_pairs = np.unique(indices_nodiag_lowfirst, axis=0)
            resp_corrs = np.array([[np.abs(spearmanr(X.iloc[:,m], y).correlation), np.abs(spearmanr(X.iloc[:,n], y).correlation)] for [m,n] in correlated_pairs])
            element_to_drop = np.argmin(resp_corrs, axis=1)
            list_to_drop = np.unique(correlated_pairs[range(element_to_drop.shape[0]),element_to_drop])
            cols_to_drop = X.columns.values[list_to_drop]

        cols_to_keep = [c for c in X.columns.values if c not in cols_to_drop]
        #print('The features used after correlation to build the model:', cols_to_keep)
        #print('Number of features retained:', len(cols_to_keep))
        self.uncorr_columns = cols_to_keep

        return self

    def transform(self, X):
        return X[self.uncorr_columns]

    def get_params(self, deep=False):
        return {'thresh': self.thresh}




#pipeline components
scaler=StandardScaler()
dropcoll=DropCollinear(0.8)
rf=RandomForestRegressor(random_state=42)
pipe=Pipeline(steps=[('dropcoll', dropcoll), ('scaler', scaler), ('rf', rf)], verbose=False)
#Parameter ranges
param_grid = {"rf__max_depth": [3, 5, 8], "rf__n_estimators": [5, 10, 15], "rf__max_features": [0.05, 0.1, 0.5], "rf__min_samples_split": [2, 3, 5]}
cv=KFold(n_splits=5)
# Optimisation
search=RandomizedSearchCV(pipe, param_grid, cv=cv, scoring='neg_mean_squared_error', return_train_score=True, verbose=0, n_iter=80, random_state=42)
print('-------Hyperparameter tuning via cross validation and training------------\n')

search.fit(X_train, y_train)
y_train_pred=search.predict(X_train)
train_rmse=sqrt(mean_squared_error(y_train, y_train_pred))

print('The best combination of hyperparameter is: \n', search.best_params_)
results=search.cv_results_
result=pd.DataFrame(results)
result.to_csv('randomForestCvResultAgain.csv', index=False) #if you want to check the cross validation result then uncomment this line
# Stores the optimum model in best_pipe
best_pipe = search.best_estimator_
print("The best pipeline details: \n", best_pipe)
print('***Training with the best hyperparameter set is over and the train RMSE is: ', train_rmse)
print('------------------------------Test-----------------------------------------\n')
y_test_pred=search.predict(X_test)
test_rmse=sqrt(mean_squared_error(y_test, y_test_pred))

#save the test prediction into a csv file
test_result=pd.DataFrame(zip(y_test, y_test_pred), columns=['y_actual', 'y_pred'])
test_result.to_csv('testSetPredictionAgain.csv', index=False)
print('***Test set prediction is done and the test RMSE is:', test_rmse)
print('-------------------------------Done-----------------------------------------')





