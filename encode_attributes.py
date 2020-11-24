import os
import pandas as pd
import numpy as np
import csv


this_folder = os.path.dirname(os.path.abspath(__file__))
train_file_name = this_folder+'/data/train.csv'
test_file_name = this_folder+'/data/assignment-test.csv'

modified_train = []
modified_test = []

with open(train_file_name) as train_file:
    for i,row in enumerate(train_file):
        row = row.replace("\n","")
        row = row.split(",")
        if i == 0: 
            modified_train.append(row)
            continue
        if row[4] == "Germany": row[4] = 0
        elif row[4] == "Spain": row[4] = 1
        else: row[4] = 2
        if row[5] ==  "Male": row[5] = 0
        else: row[5] =1
        modified_train.append(row)

with open(test_file_name) as train_file:
    for i,row in enumerate(train_file):
        row = row.replace("\n","")
        row = row.split(",")
        if i == 0: 
            modified_test.append(row)
            continue
        if row[4] == "Germany": row[4] = 0
        elif row[4] == "Spain": row[4] = 1
        else: row[4] = 2
        if row[5] ==  "Male": row[5] = 0
        else: row[5] =1
        modified_test.append(row)


with open(this_folder+"/data/modified_train.csv","w",newline="\n") as f:
    writer = csv.writer(f)
    for row in modified_train:
        writer.writerow(row)

with open(this_folder+"/data/modified_test.csv","w",newline="\n") as f:
    writer = csv.writer(f)
    for row in modified_test:
        writer.writerow(row)