# Random Forest Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

#Fitting the Random Forest Regression model to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor =RandomForestRegressor(n_estimators=300, random_state=0)
regressor.fit(X_train, y_train)

#Predicting a new result with Random Forest model
y_pred = regressor.predict(X_test)

# Visualising the Random Forest results
"""plt.scatter(X_train[:, 3], y_train, color = 'red')
plt.plot(X_train[:, 3], regressor.predict(X_train), color = 'blue')
plt.title('Startups (Random Forest Model)')
plt.xlabel('Values')
plt.ylabel('Profit')
plt.show()"""

#Visualing the Random Forest model Results (For Higher Resolution and Smoother curve)
"""X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('Truth or Bluff (Random Forest Regression Model)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()"""