from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
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

gnb = GaussianNB()
gnb.fit(X_train.toarray(), Y_train)        # input X,y for training
mnb = MultinomialNB()
mnb.fit(X_train, Y_train)

Y_test1 = gnb.predict(X_test.toarray())
output = pd.DataFrame({'id':testdata['id'],'Response':Y_test1})
output.to_csv('Bayes_gnb.csv',index = False)

Y_test2 = mnb.predict(X_test.toarray())
output = pd.DataFrame({'id':testdata['id'],'Response':Y_test2})
output.to_csv('Bayes_mnb.csv',index = False)
