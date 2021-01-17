import os
import pandas as pd
import numpy as np
import csv
import sys
import json

this_folder = os.path.dirname(os.path.abspath(__file__))
train_file_name = this_folder+'/data/onehot_train.csv'
test_file_name = this_folder+'/data/onehot_test.csv'


train_dataframe = pd.read_csv(train_file_name,usecols=['CreditScore','France', 'Germany', 'Spain', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Exited'])
test_dataframe = pd.read_csv(test_file_name,usecols=['CreditScore', 'France', 'Germany', 'Spain', 'Gender','Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary'])

x_train = train_dataframe.drop("Exited", axis=1)
y_train = train_dataframe["Exited"]



from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,VotingClassifier,BaggingClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import smote_variants

from collections import Counter

r=999

models = [
    BaggingClassifier(random_state=r),
    RandomForestClassifier(random_state=r),
    GradientBoostingClassifier(random_state=r),
    LGBMClassifier(),
    XGBClassifier(random_state=r),
    CatBoostClassifier(random_state=r,verbose = False),
]
names = [
    "Ensemble-Bagging",
    "Ensemble-Random Forest",
    "Ensemble-Gradient Boosting",
    "Light Gradient Boosting",
    "XG Boost",
    "Cat Boost",
]

samplers=[
    smote_variants.polynom_fit_SMOTE(random_state=r),
    smote_variants.ProWSyn(random_state=r),
    smote_variants.SMOTE_IPF(random_state=r),
    smote_variants.Lee(random_state=r),
    smote_variants.SMOBD(random_state=r)
]

sampler_names = [
    "smote_variants.polynom_fit_SMOTE",
    "smote_variants.ProWSyn",
    "smote_variants.SMOTE_IPF",
    "smote_variants.Lee",
    "smote_variants.SMOBD"
]


os.mkdir(this_folder+"/smotevariants")


for sampler_name, sampler in zip(sampler_names,samplers):
    x_resampled, y_resampled = sampler.sample(x_train.values, y_train.values)
    x_resampled = pd.DataFrame(x_resampled,columns=['CreditScore','France', 'Germany', 'Spain', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary'])
    y_resampled = pd.Series(y_resampled,name='Exited')
        
    os.mkdir(this_folder+"/smotevariants/"+str(sampler_name))
    outputs = {}

    for name, model in zip(names, models):
        model.fit(x_resampled, y_resampled)
        output = model.predict(test_dataframe)
        outputs[name] = output

    for model_name in outputs.keys():
        predicted = []
        predicted.append(["RowNumber","Exited"])
        for i in range(7500,10000):
            predicted.append([])
            predicted[-1].append(i)
            predicted[-1].append(outputs[model_name][i-7500])
        with open(this_folder+"/smotevariants/"+str(sampler_name)+"/"+str(model_name)+"_submission.csv","w",newline="\n") as f:
                writer = csv.writer(f)
                writer.writerows(predicted)

