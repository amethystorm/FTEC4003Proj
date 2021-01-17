from sklearn.neighbors import KNeighborsClassifier
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

#numNeighbors = [1, 3, 5, 10, 15]

k = 15
#for k in numNeighbors:        #use a for loop to 'build model' for different hyperparameter k
clf = KNeighborsClassifier(n_neighbors=k, weights='uniform', metric='minkowski', p=2)
clf.fit(X_train, Y_train)
Y_test = clf.predict(X_test)
output = pd.DataFrame({'id':testdata['id'],'Response':Y_test})
output.to_csv('KNN_k=15.csv',index = False)
