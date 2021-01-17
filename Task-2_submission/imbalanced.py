import os
import pandas as pd
import numpy as np
import csv
import sys
import json

this_folder = os.path.dirname(os.path.abspath(__file__))
train_file_name = this_folder+'/data/onehot_train.csv'
test_file_name = this_folder+'/data/onehot_test.csv'

train_dataframe = pd.read_csv(train_file_name,usecols=['CreditScore', 'Gender','France', 'Germany', 'Spain', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Exited'])
test_dataframe = pd.read_csv(test_file_name,usecols=['CreditScore', 'Gender', 'France', 'Germany', 'Spain','Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary'])

x_train = train_dataframe.drop("Exited", axis=1)
y_train = train_dataframe["Exited"]


from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,BaggingClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier



from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import SMOTENC

from imblearn.under_sampling import ClusterCentroids
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import NearMiss
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.under_sampling import RepeatedEditedNearestNeighbours
from imblearn.under_sampling import AllKNN
from imblearn.under_sampling import CondensedNearestNeighbour
from imblearn.under_sampling import OneSidedSelection
from imblearn.under_sampling import NeighbourhoodCleaningRule
from imblearn.under_sampling import InstanceHardnessThreshold

from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek

from collections import Counter
r=1001

models = [
    DecisionTreeClassifier(random_state=r),
    BaggingClassifier(random_state=r),
    RandomForestClassifier(random_state=r),
    GradientBoostingClassifier(random_state=r),
    LGBMClassifier(),
    XGBClassifier(random_state=r),
    CatBoostClassifier(random_state=r,verbose = False),
]
names = [
    "Decision Tree",
    "Ensemble-Bagging",
    "Ensemble-Random Forest",
    "Ensemble-Gradient Boosting",
    "Light Gradient Boosting",
    "XG Boost",
    "Cat Boost",
]

samplers=[
    # imbalanced-learn Over
    RandomOverSampler(random_state=r),
    SMOTE(random_state=r,k_neighbors=10), 
    ADASYN(random_state=r,n_neighbors=10),
    BorderlineSMOTE(random_state=r,k_neighbors=10),
    SMOTENC(random_state=r,categorical_features=[1,2,3,4,9,10]),
    # imbalanced-learn Under
    ClusterCentroids(random_state=r),
    RandomUnderSampler(random_state=r,replacement=True),
    NearMiss(version=1,n_neighbors=10),
    EditedNearestNeighbours(),
    RepeatedEditedNearestNeighbours(),
    AllKNN(),
    CondensedNearestNeighbour(random_state=r),
    OneSidedSelection(random_state=r),
    NeighbourhoodCleaningRule(),
    InstanceHardnessThreshold(random_state=r,estimator=LogisticRegression(solver='lbfgs', multi_class='auto')),
    # imbalanced-learn Combine Over and Under
    SMOTEENN(random_state=r),
    SMOTETomek(random_state=r)
]

sampler_names = [
    "RandomOverSampler",
    "SMOTE", 
    "ADASYN",
    "BorderlineSMOTE",
    "SMOTENC",
    "ClusterCentroids",
    "RandomUnderSampler",
    "NearMiss",
    "EditedNearestNeighbours",
    "RepeatedEditedNearestNeighbours",
    "AllKNN",
    "CondensedNearestNeighbour",
    "OneSidedSelection",
    "NeighbourhoodCleaningRule",
    "InstanceHardnessThreshold",
    "SMOTEENN",
    "SMOTETomek"
]

os.mkdir(this_folder+"/imbalanced")

for sampler_name, sampler in zip(sampler_names,samplers):
    x_resampled, y_resampled = sampler.fit_resample(x_train, y_train)

    os.mkdir(this_folder+"/imbalanced/"+str(sampler_name))
    
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
        with open(this_folder+"/imbalanced/"+str(sampler_name)+"/"+str(model_name)+"_submission.csv","w",newline="\n") as f:
                writer = csv.writer(f)
                writer.writerows(predicted)




