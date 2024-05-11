import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split


df=sns.load_dataset('mpg')


df.isnull().sum()
df.dropna(inplace=True)


X=df[['displacement',	'horsepower',	'weight',	'acceleration']]
y=df.mpg
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42)


#from ctypes import LibraryLoader
from sklearn.linear_model import LinearRegression


model=LinearRegression()
model.fit(X_train,y_train)


model.score(X_test,y_test)

df.isnull().sum()


from sklearn.tree import DecisionTreeRegressor
model2=DecisionTreeRegressor(criterion="poisson",random_state=0)  #criterion{“squared_error”, “friedman_mse”, “absolute_error”, “poisson”}, default=”squared_error”
model2.fit(X_train,y_train)


model2.score(X_test,y_test)


import pickle
filename = 'mpg_regression.sav'
pickle.dump(model, open(filename, 'wb'))


