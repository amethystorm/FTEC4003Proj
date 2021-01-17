import os
import pandas as pd
import numpy as np
import csv
import sys
import json
import time


this_folder = os.path.dirname(os.path.abspath(__file__))
train_file_name = this_folder+'/data/onehot_train.csv'
test_file_name = this_folder+'/data/onehot_test.csv'

# Use only the meaningful attributes
train_attrs = list(pd.read_csv(train_file_name, nrows =1))
train_dataframe = pd.read_csv(train_file_name,usecols=['CreditScore','France', 'Germany', 'Spain', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Exited'])
test_attrs = list(pd.read_csv(test_file_name, nrows =1))
test_dataframe = pd.read_csv(test_file_name,usecols=['CreditScore', 'France', 'Germany', 'Spain', 'Gender','Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary'])


x_train = train_dataframe.drop("Exited", axis=1)
y_train = train_dataframe["Exited"]


from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

r=999
Xs = [1,2,3,5,10]
# Xs = [1.5,1.6,1.7,1.8,1.9,2.1,2.2,2.3]

names = [
    "Ensemble-RandomForest",
    "LightGradientBoosting",
    "CatBoost"
]

for x in Xs:
    models = [
        RandomForestClassifier(random_state=r,class_weight={0:1,1:x}),
        LGBMClassifier(random_state=r,class_weight={0:1,1:x}),
        CatBoostClassifier(random_state=r,verbose = False,class_weights={0:1,1:x},loss_function='Logloss',eval_metric="F1")
    ]
    os.mkdir(this_folder+"/class_weights_"+str(x))

    outputs = {}
    for name, model in zip(names, models):
        model.fit(x_train, y_train)
        output = model.predict(test_dataframe)
        outputs[name] = output



    for model_name in outputs.keys():
        predicted = []
        predicted.append(["RowNumber","Exited"])
        for i in range(7500,10000):
            predicted.append([])
            predicted[-1].append(i)
            predicted[-1].append(outputs[model_name][i-7500])
        with open(this_folder+"/class_weights_"+str(x)+"/"+str(model_name)+"_submission.csv","w",newline="\n") as f:
                writer = csv.writer(f)
                writer.writerows(predicted)
