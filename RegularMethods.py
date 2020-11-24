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

#Decision Tree      around 0.5

# from sklearn.tree import DecisionTreeClassifier
# decision_tree = DecisionTreeClassifier(criterion="gini",splitter="best",max_depth=10)
# decision_tree = decision_tree.fit(x_train,y_train)
# decision_tree_output = decision_tree.predict_proba(test_dataframe)



# KNN               around 0.15

# from sklearn.neighbors import KNeighborsClassifier
# num_of_neighbors = [5]
# for k in num_of_neighbors:
#     k_nearest_neighbors = KNeighborsClassifier(n_neighbors=k,weights="distance",metric='minkowski',p=2)
#     k_nearest_neighbors.fit(x_train,y_train)
#     k_nearest_neighbors_output = k_nearest_neighbors.predict_proba(test_dataframe)



# Bayes

# from sklearn.naive_bayes import GaussianNB
# from sklearn.naive_bayes import MultinomialNB

# gnb = GaussianNB()
# gnb.fit(x_train,y_train)
# gnb_output = gnb.predict_proba(test_dataframe) # around 0.1

# mnb = MultinomialNB()
# mnb.fit(x_train,y_train)
# mnb_output = mnb.predict_proba(test_dataframe) # around 0.3



# Logistic Regression       around 0.07

# from sklearn.linear_model import LogisticRegression
# logistic_regression=LogisticRegression()
# logistic_regression.fit(x_train,y_train)
# logistic_regression_output = logistic_regression.predict_proba(test_dataframe)



# SVM           Error

# from sklearn.svm import SVC
# support_vector_machine = SVC(C=0.000001, gamma='auto',probability=True)
# support_vector_machine.fit(x_train,y_train)
# support_vector_machine_output = support_vector_machine.predict_proba(test_dataframe)
# print(support_vector_machine_output)



# Multilayer Neural Network     0.11

# from sklearn.neural_network import MLPClassifier
# MNN = MLPClassifier(hidden_layer_sizes=(100,100,100),solver='adam',max_iter=10)
# MNN.fit(x_train,y_train)
# MNN_output = MNN.predict(test_dataframe)

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
        # predicted[-1].append(decision_tree_output[i-1][1]) # decision tree
        # predicted[-1].append(k_nearest_neighbors_output[i-1][1]) # knn
        # predicted[-1].append(gnb_output[i-1][1]) # Gaussian Naive Bayes
        # predicted[-1].append(mnb_output[i-1][1]) # Multinomia Naive Bayes
        # predicted[-1].append(logistic_regression_output[i-1][1]) # Logistic Regression
        # predicted[-1].append(support_vector_machine_output[i-1][1]) # Support Vector Machine
        # predicted[-1].append(MNN_output[i-1]) # Multilayer Neural Network

with open(this_folder+"/submission.csv","w",newline="\n") as f:
        writer = csv.writer(f)
        writer.writerows(predicted)