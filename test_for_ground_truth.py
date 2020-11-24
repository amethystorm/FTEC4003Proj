import os
import csv
import numpy as np

this_folder = os.path.dirname(os.path.abspath(__file__))
true_predicts=[]
false_predicts=[]

with open(this_folder+"/groundtruth.txt") as f:
    ground_truth=f.read()

ground_truth=ground_truth.split(",")
ground_truth[0]=ground_truth[0].replace("[","")
ground_truth[-1]=ground_truth[-1].replace("]","")

submission_data=[]

with open(this_folder+"/submission.csv") as testfile:
    csv_reader=csv.reader(testfile)
    for i,row in enumerate(csv_reader):
        if i==0:continue
        submission_data.append(float(row[1]))


for i,label in enumerate(ground_truth):
    if int(label)==1:true_predicts.append(submission_data[i])
    if int(label)==0:false_predicts.append(submission_data[i])

print(min(true_predicts),max(false_predicts))

#print(true_predicts)