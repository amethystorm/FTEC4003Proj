from sklearn import tree                   # import the packages we need
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.feature_extraction import DictVectorizer

import os
this_folder = os.path.dirname(os.path.abspath(__file__))
traindata = pd.read_csv(this_folder+"/insurance-train.csv")
testdata = pd.read_csv(this_folder+"/insurance-test.csv")

X_train = traindata[['Gender','Age','Driving_License','Region_Code','Previously_Insured','Vehicle_Age','Vehicle_Damage','Annual_Premium','Policy_Sales_Channel','Vintage']]
Y_train = traindata['Response']
X_test = testdata[['Gender','Age','Driving_License','Region_Code','Previously_Insured','Vehicle_Age','Vehicle_Damage','Annual_Premium','Policy_Sales_Channel','Vintage']]

vec = DictVectorizer()
X_train = vec.fit_transform(X_train.to_dict(orient="record"))
X_test = vec.transform(X_test.to_dict(orient="record"))

clf = tree.DecisionTreeClassifier(max_depth = None)
clf = clf.fit(X_train,Y_train)     # input X,y for training
Y_test = clf.predict(X_test)
output = pd.DataFrame({'id':testdata['id'],'Response':Y_test})
output.to_csv('DecisionTree_size=max.csv',index = False)


