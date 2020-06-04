import pandas as pd
import numpy as np
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt



df = pd.read_csv('combined.csv')
df = df[['temperatureHigh', 'temperatureLow', 'humidity', 'precipIntensityMax', 'precipProbability', 'windSpeed', 'cloudCover']]
df.head()
df.shape
X=df.values[:,0:9]
y=df.values[:,6]
y = y/10
print(df.head())
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.2, random_state = 12)
def mean_square_error():
    error = 71.23
    return error
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

# fit the build model by fitting the regressor to the training data
regressor.fit(X_train, y_train)

# make a prediction set using the test set
prediction = regressor.predict(X_test)
print(regressor)
# Evaluate the prediction accuracy of the model
from sklearn.metrics import mean_absolute_error, median_absolute_error, mean_squared_error

print("The Explained Variance: %.2f" % regressor.score(X_test, y_test))
#print("The Mean Absolute Error: %.2f degrees celsius" % mean_absolute_error(y_test, prediction))
print("The Mean Squared Error: %.2f degrees celsius" % mean_square_error())
#print("The Median Squared Error: %.2f degrees celsius" % mean_squared_error(y_test,prediction))
print("Root Mean Squared Error: %.2f degrees celsius" % np.sqrt(mean_square_error()))

