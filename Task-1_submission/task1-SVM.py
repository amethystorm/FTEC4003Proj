from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,BaggingClassifier
from sklearn.feature_extraction import DictVectorizer
import pandas as pd
import os
import sys
from joblib import dump, load
import time
from contextlib import contextmanager

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} done in {:.0f}s".format(title, time.time() - t0))

this_folder = os.path.dirname(os.path.abspath(__file__))
traindata = pd.read_csv(this_folder+"/insurance-train.csv")
testdata = pd.read_csv(this_folder+"/insurance-test.csv")

xtrain = traindata[['Gender','Age','Driving_License','Region_Code','Previously_Insured','Vehicle_Age','Vehicle_Damage','Annual_Premium','Policy_Sales_Channel','Vintage']]
ytrain = traindata['Response']
x_predict = testdata[['Gender','Age','Driving_License','Region_Code','Previously_Insured','Vehicle_Age','Vehicle_Damage','Annual_Premium','Policy_Sales_Channel','Vintage']]

r = 999

vec = DictVectorizer()
xtrain = vec.fit_transform(xtrain.to_dict(orient="record"))

x_predict = vec.transform(x_predict.to_dict(orient="record"))

Cs = [0.001,0.01, 0.1, 1, 10, 100]


names = [
    # "DecisionTree",
    # "KNeighbors",
    # "GaussianNB",
    # "MultinomialNB",
    "SVC",
    # "Ensemble-Bagging",
    # "Ensemble-RandomForest",
    # "Ensemble-GradientBoosting",
]

os.mkdir(this_folder+"/SVM_sigmoid")

for c in Cs:

    models = [
        # DecisionTreeClassifier(random_state=r),
        # KNeighborsClassifier(n_jobs=-1),
        # GaussianNB(),
        # MultinomialNB(),
        SVC(random_state=r,C=c,kernel='sigmoid'),  # 'linear', 'poly', 'rbf', 'sigmoid'
        # BaggingClassifier(random_state=r,n_jobs=-1),
        # RandomForestClassifier(random_state=r,n_jobs=-1),
        # GradientBoostingClassifier(random_state=r),
    ]
    for name, model in zip(names, models):
        with timer(">Model training"):
            model.fit(xtrain, ytrain)
            output = model.predict(x_predict)
            dump(model, this_folder+'/SVM_sigmoid/'+str(c)+'.joblib')




    
    