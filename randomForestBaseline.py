# -*- coding: utf-8 -*-
"""
Created on Tuesday September  3 11:34:02 2021

@author: Manajit
"""
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from math import sqrt
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import argparse

def randomForest(input_csv, output_csv):
    df=pd.read_csv(input_csv)
    X=df.iloc[:, :-1]
    y=df.iloc[:, -1]
    X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=42)
    regr=RandomForestRegressor()
    regr.fit(X_train, y_train)
    y_pred=regr.predict(X_test)
    mse=mean_squared_error(y_test, y_pred)
    rmse=sqrt(mse)
    print(rmse)
    result = pd.DataFrame(list(zip(X_test.index, y_test, y_pred)), columns =['sampleIndex', 'y_test', 'y_pred'])
    result.to_csv(output_csv, index=False)
    return y_test, y_pred


def plot(y_test, y_pred, figName):
    plt.scatter(y_test, y_pred)
    plt.xlabel('y_test')
    plt.ylabel('y_pred')
    plt.savefig(figName)


if __name__=="__main__":
    y_test, y_pred=randomForest('final-260-137-borderline2-syn.csv', 'final-260-137-borderline2-syn_output.csv')
    plot(y_test, y_pred, 'final-260-137-borderline2-syn.csv.png')





