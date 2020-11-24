import os
import pandas as pd
import numpy as np
import csv


this_folder = os.path.dirname(os.path.abspath(__file__))
train_file_name = this_folder+'/data/modified_train.csv'
test_file_name = this_folder+'/data/modified_test.csv'

# Use only the meaningful attributes
train_attrs = list(pd.read_csv(train_file_name, nrows =1))
train_dataframe = pd.read_csv(train_file_name,usecols=train_attrs[-11:])
test_attrs = list(pd.read_csv(test_file_name, nrows =1))
test_dataframe = pd.read_csv(test_file_name,usecols=test_attrs[-10:])
# print(useful_cols[-11:])
# print(dataframe.head())
# print(dataframe.tail())


# Split train & val datasets
# val_dataframe = dataframe.sample(frac=0.2, random_state=1337)
# train_dataframe = dataframe.drop(val_dataframe.index)

x_train = train_dataframe.drop("Exited", axis=1)
y_train = train_dataframe["Exited"]

#Decision Tree
from sklearn.tree import DecisionTreeClassifier
decision_tree = DecisionTreeClassifier(criterion="gini",splitter="best",max_depth=10)
decision_tree = decision_tree.fit(x_train,y_train)
decision_tree_output = decision_tree.predict_proba(test_dataframe)

predicted = []
with open(this_folder+"/data/assignment-test.csv") as testfile:
    for i, row in enumerate(testfile):
        if i==0:
            predicted.append(["RowNumber","Exited"])
            continue
        predicted.append([])
        row=row.split(",")
        row[-1]=row[-1].replace("\n","")
        predicted[-1].append(row[0])
        predicted[-1].append(decision_tree_output[i-1][1])


with open(this_folder+"/submission.csv","w",newline="\n") as f:
        writer = csv.writer(f)
        writer.writerows(predicted)