# Linear-Regression
### Code
```
from sklearn import preprocessing, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


# 資料輸入
df = pd.read_csv('./dataset/housing.csv', header = None, delim_whitespace=True)
# 答案取出
y = df[13]
x = df.drop(13, axis = 1)


# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1)


# Normalization
scaler = preprocessing.StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# Model Select
model = linear_model.LinearRegression()
model.fit(x_train, y_train)

# Predict
y_pred = model.predict(x_test)

print('Cofficient : {}'.format(model.coef_))

print('Mean squared error : {}'.format(mean_squared_error(y_test, y_pred)))

print('Variance score : {}'.format(r2_score(y_test, y_pred)))
```
# Polynomial Regression
```
import numpy as np
import pandas as pd
from sklearn import linear_model, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

# 匯入檔案
df = pd.read_csv('./dataset/winequality-red.csv')

# 處理answer and data
y = df['quality']
x = df.drop('quality', axis = 1)

# 產生degree 為 2 的feature
poly = PolynomialFeatures(degree = 2).fit(x)
x = poly.transform(x)

# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 1)

# Normalization
scaler = preprocessing.StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# Select model
model = linear_model.LinearRegression()
model.fit(x_train, y_train)

# Predict
y_pred = model.predict(x_test)

# 查看係數
print('The coefficient : {}\n'.format(model.coef_))
print('Mean squared error : {}'.format(mean_squared_error(y_test, y_pred)))
print('Variance score : {}'.format(r2_score(y_test, y_pred)))
```
# Logistic Regression
```
import numpy as np
import pandas as pd
from sklearn import preprocessing, linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


pima = pd.read_csv('./dataset/pima-indians-diabetes.csv')

#x = pima[['pregnant', 'insulin', 'bmi', 'age']]
y = pima['label']
x = pima.drop(['label'], axis = 1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 1)

scaler = preprocessing.StandardScaler().fit(x_train)

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

model = LogisticRegression()
model.fit(x_train, y_train)

print(model.coef_)

print(model.intercept_)

y_pred = model.predict(x_test)

print(y_pred)
accuracy = model.score(x_test, y_test)

print(accuracy)
```
