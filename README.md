# DATA-SCIENCE-FINAL-ASSIGNMENT


import pandas as pd
import numpy as np

data = pd.read_csv("heart.csv")
data

data.isnull().sum()

data.describe()

data.nunique()

data['age'].unique()

import seaborn as sns

corelation = data.corr()

sns.heatmap(corelation, xticklabels=corelation.columns, yticklabels=corelation.columns ,annot=True)

sns.heatmap(data/np.sum(data), annot=True, fmt='.2%', cmap='Blues')

X=data.iloc[:,0:13]
X

sns.pairplot(X)

sns.relplot(x="sex", y="target", data=data)

sns.relplot(x="age", y="target", data=data)

sns.distplot(data["sex"])

sns.distplot(data["age"])

sns.catplot(x="age", kind='box', data=data)

Y = data.iloc[:,13]
Y

import sklearn.preprocessing as pre_process

standered_scalling = pre_process.StandardScaler()

X.columns

import sklearn.preprocessing as pre_process
import numpy as np
ordinal_encoding=pre_process.OrdinalEncoder()
standered_scalling=pre_process.StandardScaler()
min_max_scaler = pre_process.MinMaxScaler()

from sklearn.compose import make_column_transformer
transform_x=make_column_transformer((min_max_scaler , ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
       'exang', 'oldpeak', 'slope', 'ca', 'thal']))
       
transform_x
    
processed_x=transform_x.fit_transform(X)
processed_x   

from sklearn import preprocessing
from sklearn import utils

#convert y values to categorical values
lab = preprocessing.LabelEncoder()
y_transformed = lab.fit_transform(Y)


y_transformed.reshape(-1,1)

processed_y=standered_scalling.fit_transform(np.c_[Y]).reshape(-1,1)
processed_y

y_transformed

processed_x







from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y = train_test_split(processed_x,y_transformed)

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
processed_x, y_transformed = make_classification(n_samples=1000, n_features=4,n_informative=2, n_redundant=0,random_state=0, shuffle=False)
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(processed_x, y_transformed)


clf = RandomForestClassifier(n_estimators = 1025)


clf.fit(processed_x, y_transformed)


y_pred = clf.predict(processed_x)


from sklearn import metrics
print()


ans = metrics.accuracy_score(y_transformed, y_pred)
ans = ans*100
print("ACCURACY OF THE MODEL: ",ans,"%")












from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd



seed = 8
kfold = model_selection.KFold(n_splits = 3,random_state = seed, shuffle=True)


base_cls = DecisionTreeClassifier()


num_trees = 1024


model = BaggingClassifier(base_estimator = base_cls,n_estimators = num_trees,random_state = seed)

results = model_selection.cross_val_score(model, processed_x, y_transformed, cv = kfold)
print("accuracy :")
print(results.mean())





[Data Science Final Assignment.pdf](https://github.com/Ibtasam-Ahmad/DATA-SCIENCE-FINAL-ASSIGNMENT/files/8958672/Data.Science.Final.Assignment.pdf)


