import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp
import warnings
import os
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
import datetime
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import pickle
data=pd.read_csv('data.csv')
#print(data.head())
print(data.describe() )     #description of dataset 
print(data.info())
print(data.shape       )   #569 rows and 33 columns
print(data.columns )    #displaying the columns of dataset
print(data.dtypes)
print(data.isnull().sum())
data.drop('Unnamed: 32', axis = 1, inplace = True)
print(data)
x = data.drop(columns = ['id','diagnosis'])
y = data['diagnosis']
print("xhead is**********************************************************")
print(x.head())
#train_test_splitting of the dataset
from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
print(len(x_train))
print(len(x_test))
print(len(y_train))
print(len(y_test))

from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.ensemble import AdaBoostClassifier
adb = AdaBoostClassifier(base_estimator = None)
adb.fit(x_train,y_train)
y_pred=adb.predict(x_test)
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,mean_squared_error,r2_score
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
print("Training Score: ",adb.score(x_train,y_train)*100)
print(accuracy_score(y_test,y_pred)*100)
filename = 'finalized_model.sav'
pickle.dump(adb, open(filename, 'wb'))
