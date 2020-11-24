import os
import csv
import json

this_folder = os.path.dirname(os.path.abspath(__file__))

train_data = []
test_data = []
test_ground_truth = []

with open(this_folder+"/groundtruth.txt","r") as f:
    ground_truth=f.read()

ground_truth=ground_truth.split(",")
ground_truth[0]=ground_truth[0].replace("[","")
ground_truth[-1]=ground_truth[-1].replace("]","")

with open(this_folder+"/train.csv","r") as f:
    csv_train_data = csv.reader(f)
    for i,data in enumerate(csv_train_data):
        if i == 0: attributes = data
        else:train_data.append(data)

with open(this_folder+"/assignment-test.csv","r") as f:
    csv_test_data = csv.reader(f)
    for i,data in enumerate(csv_test_data):
        if i == 0: continue
        else:test_data.append(data)

#Observe the distribution

distribution = {"train":{},"test":{}}

#Gender
distribution["train"]["Gender"] = {"Male":[0,0],"Female":[0,0]}
distribution["test"]["Gender"] = {"Male":[0,0],"Female":[0,0]}
for i,row in enumerate(train_data):
    if int(row[-1])==0:distribution["train"]["Gender"][row[5]][0]+=1
    if int(row[-1])==1:distribution["train"]["Gender"][row[5]][1]+=1

for i,row in enumerate(test_data):
    if int(ground_truth[i])==0:distribution["test"]["Gender"][row[5]][0]+=1
    if int(ground_truth[i])==1:distribution["test"]["Gender"][row[5]][1]+=1

#Geography
distribution["train"]["Geography"] = {"Germany":[0,0],"France":[0,0],"Spain":[0,0]}
distribution["test"]["Geography"] = {"Germany":[0,0],"France":[0,0],"Spain":[0,0]}
for i,row in enumerate(train_data):
    if int(row[-1])==0:distribution["train"]["Geography"][row[4]][0]+=1
    if int(row[-1])==1:distribution["train"]["Geography"][row[4]][1]+=1

for i,row in enumerate(test_data):
    if int(ground_truth[i])==0:distribution["test"]["Geography"][row[4]][0]+=1
    if int(ground_truth[i])==1:distribution["test"]["Geography"][row[4]][1]+=1


with open(this_folder + '/distribution.json', 'w') as outfile:
    json.dump(distribution, outfile, indent=2)

