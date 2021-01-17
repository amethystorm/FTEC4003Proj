import os
import pandas as pd
import numpy as np
import csv
import sys
import json

this_folder = os.path.dirname(os.path.abspath(__file__))
train_file_name = this_folder+'/data/onehot_train.csv'
test_file_name = this_folder+'/data/onehot_test.csv'

# 'CreditScore','France', 'Germany', 'Spain', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Exited'
train_dataframe = pd.read_csv(train_file_name,usecols=['CreditScore','France', 'Germany', 'Spain', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Exited'])
test_dataframe = pd.read_csv(test_file_name,usecols=['CreditScore', 'France', 'Germany', 'Spain', 'Gender','Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary'])

x_train = train_dataframe.drop("Exited", axis=1)
y_train = train_dataframe["Exited"]


from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.under_sampling import NeighbourhoodCleaningRule

import smote_variants
from collections import Counter


r=999

outputs = []

# Model 1 XGBst + ProWSyn 0.63217


# x_resampled, y_resampled = smote_variants.ProWSyn(random_state=r).sample(x_train.values, y_train.values)
# x_resampled = pd.DataFrame(x_resampled,columns=['CreditScore','France', 'Germany', 'Spain', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary'])
# y_resampled = pd.Series(y_resampled,name='Exited')

# model_1 = XGBClassifier(random_state=r,learning_rate=0.3,max_depth=4,n_estimators=50,subsample=1)
# model_1.fit(x_resampled,y_resampled)
# model_1_output_prob = model_1.predict_proba(test_dataframe)

# model_1_output = []
# for i, prob in enumerate(model_1_output_prob):
#     if float(model_1_output_prob[i][0]) > 0.55: model_1_output.append(0)
#     else: model_1_output.append(1)

# outputs.append(model_1_output)



# Model 2 LGBst + Class_weight 0.63512

model_2 = LGBMClassifier(random_state=r,boosting_type='gbdt',class_weight={0: 1, 1: 2}, n_estimators=45)
model_2.fit(x_train,y_train)
model_2_output_prob = model_2.predict_proba(test_dataframe)

model_2_output = []
for i, prob in enumerate(model_2_output_prob):
    if float(model_2_output_prob[i][0]) > 0.473: model_2_output.append(0)
    else: model_2_output.append(1)

outputs.append(model_2_output)



# # Model 3 GBst + NeighbourhoodCleaningRule
# x_resampled, y_resampled = NeighbourhoodCleaningRule().fit_resample(x_train, y_train)

# model_3 = GradientBoostingClassifier(random_state=1001)
# model_3.fit(x_resampled,y_resampled)
# model_3_output_prob = model_3.predict_proba(test_dataframe)

# model_3_output = []
# for i, prob in enumerate(model_3_output_prob):
#     if float(model_3_output_prob[i][0]) > 0.42: model_3_output.append(0)
#     else: model_3_output.append(1)

# outputs.append(model_3_output)

# final_preds = []

# for i in range(0,2500):
#     pred = 0
#     for model_output in outputs:
#         if model_output[i] == 1:pred += 1
#     if pred/2 < 0.49: final_preds.append(0)
#     else: final_preds.append(1)

for model_pred in outputs:
    predicted = []
    predicted.append(["RowNumber","Exited"])
    for i in range(7500,10000):
        predicted.append([])
        predicted[-1].append(i)
        predicted[-1].append(model_pred[i-7500])
    with open(this_folder+"/submission_2.csv","w",newline="\n") as f:
            writer = csv.writer(f)
            writer.writerows(predicted)

