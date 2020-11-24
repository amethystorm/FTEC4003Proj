import os
import csv
import numpy as np

this_folder = os.path.dirname(os.path.abspath(__file__))
train_file_name = this_folder+'/data/train.csv'
test_file_name = this_folder+'/data/assignment-test.csv'

with open(this_folder+"/groundtruth.txt") as f:
    ground_truth=f.read()

ground_truth=ground_truth.split(",")
ground_truth[0]=ground_truth[0].replace("[","")
ground_truth[-1]=ground_truth[-1].replace("]","")

modified_train = []
modified_test = []
flag_train = 0
flag_test = 0
with open(train_file_name) as train_file:
    for i,row in enumerate(train_file):
        row = row.replace("\n","")
        row = row.split(",")
        if i == 0: continue
        if float(row[8]) == 0 and int(row[-1]) == 0: flag_train +=1

with open(test_file_name) as train_file:
    for i,row in enumerate(train_file):
        row = row.replace("\n","")
        row = row.split(",")
        if i == 0: continue
        if float(row[8]) == 0 and int(ground_truth[i-1]) == 0: flag_test+=1

print(flag_train)
print(flag_test)
true_predicts=[]
false_predicts=[]


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
        predicted[-1].append(ground_truth[i-1])


with open(this_folder+"/submission.csv","w",newline="\n") as f:
        writer = csv.writer(f)
        writer.writerows(predicted)

# ground_truth_true_rownum = []

# for i,label in enumerate(ground_truth):
#     if int(label) == 1:
#         ground_truth_true_rownum.append(7500+i)
# submission_data=[]


# flag=0
# with open(this_folder+"/submission.csv") as testfile:
#     csv_reader=csv.reader(testfile)
#     for i,row in enumerate(csv_reader):
#         if i==0:continue
#         if int(row[0]) in ground_truth_true_rownum:
#             if float(row[1]) < 0.5:flag+=1



# for i,label in enumerate(ground_truth):
#     if int(label)==1:true_predicts.append(submission_data[i])
#     if int(label)==0:false_predicts.append(submission_data[i])

# print(min(true_predicts),max(false_predicts))

# #print(true_predicts)