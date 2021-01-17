import os
import pandas as pd
import numpy as np
import csv
import sys
import json

this_folder = os.path.dirname(os.path.abspath(__file__))
train_file_name = this_folder+'/data/onehot_train.csv'
test_file_name = this_folder+'/data/onehot_test.csv'

# Use only the meaningful attributes
train_attrs = list(pd.read_csv(train_file_name, nrows =1))
#['RowNumber', 'CustomerId', 'Surname', 'CreditScore', 'France', 'Germany', 'Spain', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Exited']
train_dataframe = pd.read_csv(train_file_name,usecols=['CreditScore', 'Gender','France', 'Germany', 'Spain', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Exited'])
test_attrs = list(pd.read_csv(test_file_name, nrows =1))
#['RowNumber', 'CustomerId', 'Surname', 'CreditScore', 'France', 'Germany', 'Spain', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
test_dataframe = pd.read_csv(test_file_name,usecols=['CreditScore', 'Gender', 'France', 'Germany', 'Spain','Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary'])

x_train = train_dataframe.drop("Exited", axis=1)
y_train = train_dataframe["Exited"]


from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,BaggingClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.ensemble import BalancedBaggingClassifier, BalancedRandomForestClassifier, RUSBoostClassifier

r=999

models = [
    DecisionTreeClassifier(random_state=r),
    KNeighborsClassifier(),
    GaussianNB(),
    MultinomialNB(),
    LogisticRegression(random_state=r),
    SVC(random_state=r,kernel='sigmoid'),
    MLPClassifier(random_state=r),
    BaggingClassifier(random_state=r),
    RandomForestClassifier(random_state=r),
    GradientBoostingClassifier(random_state=r),
    LGBMClassifier(),
    XGBClassifier(random_state=r),
    CatBoostClassifier(random_state=r,verbose=False),
    BalancedBaggingClassifier(random_state=r), 
    BalancedRandomForestClassifier(random_state=r), 
    RUSBoostClassifier(random_state=r)
]
names = [
    "DecisionTree",
    "KNeighbors",
    "GaussianNB",
    "MultinomialNB",
    "LogisticRegression",
    "SVC",
    "MLPClassifier",
    "Ensemble-Bagging",
    "Ensemble-RandomForest",
    "Ensemble-GradientBoosting",
    "LightGradientBoosting",
    "XGBoost",
    "CatBoost",
    "BalancedBagging", 
    "BalancedRandomForest", 
    "RUSBoost"
]

outputs = {}

for name, model in zip(names, models):
    model.fit(x_train, y_train)
    output = model.predict(test_dataframe)
    outputs[name] = output

os.mkdir(this_folder+"/default_models_results")



for model_name in outputs.keys():
    predicted = []
    predicted.append(["RowNumber","Exited"])
    for i in range(7500,10000):
        predicted.append([])
        predicted[-1].append(i)
        predicted[-1].append(outputs[model_name][i-7500])
    with open(this_folder+"/default_models_results/"+str(model_name)+"_submission.csv","w",newline="\n") as f:
            writer = csv.writer(f)
            writer.writerows(predicted)
