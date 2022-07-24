#pip install mljar-supervised

import pandas as pd 
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from supervised.automl import AutoML
from supervised.preprocessing.eda import EDA


df_train = pd.read_csv("/content/train.csv")
df_test = pd.read_csv("/content/test.csv")
X_train,y_train = df_train.drop(['Outcome'],axis=1),df_train['Outcome']
X_test,y_test = df_test.drop(['Outcome'],axis=1),df_test['Outcome']

EDA.extensive_eda(X_train,y_train,save_path="/content/mljar-supervised/")

a = AutoML(mode='Perform',total_time_limit=10)
a.fit(X_train,y_train)
