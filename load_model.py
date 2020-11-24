import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import csv
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow.keras.layers.experimental.preprocessing import CategoryEncoding


this_folder = os.path.dirname(os.path.abspath(__file__))
model = keras.models.load_model(this_folder)
attr = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
predicted = []

with open(this_folder+"/assignment-test.csv") as testfile:
    for i, row in enumerate(testfile):
        if i==0:
            predicted.append(["RowNumber","Exited"])
            continue
        predicted.append([])
        row=row.split(",")
        row[-1]=row[-1].replace("\n","")
        predicted[-1].append(row[0])
        useful_data=row[-10:]
        sample = {}
        for j,value in enumerate(useful_data):
            if attr[j] in ['CreditScore', 'Age', 'Tenure', 'NumOfProducts', 'HasCrCard', 'IsActiveMember']:sample[attr[j]]=int(value)
            elif attr[j] in ['Balance','EstimatedSalary']: sample[attr[j]]=float(value)
            else: sample[attr[j]]=value
        input_dict = {name: tf.convert_to_tensor([value]) for name, value in sample.items()}
        predictions = model.predict(input_dict)
        if predictions[0][0] < 0.5: predicted[-1].append(0)
        else: predicted[-1].append(1)

with open(this_folder+"/submission.csv","w",newline="\n") as f:
        writer = csv.writer(f)
        writer.writerows(predicted)