import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import r2_score
import urllib.request


pd.set_option('display.width', 0)


filename = 'FuelConsumption.csv'

if not os.path.exists(filename):
    url = 'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/FuelConsumptionCo2.csv'
    urllib.request.urlretrieve(url, filename)

df = pd.read_csv(filename)

df_described = df.describe()
print('\n')
print(df_described)

cdf = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]

##
# %% code cell
fig, axs = plt.subplots(2, 4)
viz = cdf[['CYLINDERS', 'ENGINESIZE', 'CO2EMISSIONS', 'FUELCONSUMPTION_COMB']]
viz.hist(ax=(axs[0][0], axs[0][1], axs[0][2], axs[0][3]))

##
axs[1][0].scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS,  color='blue')
axs[1][0].set_xlabel("FUELCONSUMPTION_COMB")
axs[1][0].set_ylabel("Emission")

##
axs[1][1].scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue')
axs[1][1].set_xlabel("Engine size")
axs[1][1].set_ylabel("Emission")

##
axs[1][2].scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS,  color='red')
axs[1][2].set_xlabel("CYLINDERS")
axs[1][2].set_ylabel("Emission")


##
# Creating train and test dataset

msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

# Simple Regression Model

axs[1][3].set_title('Training Data')
axs[1][3].scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
axs[1][3].set_xlabel("Engine size")
axs[1][3].set_ylabel("Emission")
plt.show(block=False)


# Modeling
print('\n\nUsing sklearn package to model data.')


regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(train_x, train_y)
# The coefficients
print('\nCoefficients: ', regr.coef_)
print('Intercept: ', regr.intercept_)

plt.figure(6)

plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")


test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_hat = regr.predict(test_x)

print("\nMean absolute error: %.2f" % np.mean(np.absolute(test_y_hat - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_hat - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y_hat, test_y))

plt.show()
