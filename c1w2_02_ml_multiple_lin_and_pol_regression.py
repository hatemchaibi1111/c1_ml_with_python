import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import urllib.request

from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score


filename = 'FuelConsumption.csv'

if not os.path.exists(filename):
    url = ('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/'
           'FuelConsumptionCo2.csv')
    urllib.request.urlretrieve(url, filename)

df = pd.read_csv(filename)

cdf = df[['ENGINESIZE',
          'CYLINDERS',
          'FUELCONSUMPTION_CITY',
          'FUELCONSUMPTION_HWY',
          'FUELCONSUMPTION_COMB',
          'CO2EMISSIONS']]

print('\n\nMultiple linear regression\n')


fig, axs = plt.subplots(2, 2)
axs[0][0].scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue')
axs[0][0].set_xlabel("Engine size")
axs[0][0].set_ylabel("Emission")

plt.show(block=False)

msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

axs[0][1].scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
axs[0][1].set_title('Train Data')
axs[0][1].set_xlabel("Engine size")
axs[0][1].set_ylabel("Emission")

regr = linear_model.LinearRegression()
x = np.asanyarray(train[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']])
y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(x, y)
# The coefficients
print('Coefficients: ', regr.coef_)
print('Intercept: ', regr.intercept_)


y_hat = regr.predict(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
x = np.asanyarray(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
y = np.asanyarray(test[['CO2EMISSIONS']])

print("Residual sum of squares: %.2f" % np.mean((y_hat - y) ** 2))

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(x, y))

##
# Practice
regr = linear_model.LinearRegression()
x = np.asanyarray(train[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY']])
y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(x, y)

y_hat = regr.predict(test[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY']])
x = np.asanyarray(test[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY']])
y = np.asanyarray(test[['CO2EMISSIONS']])

print("Residual sum of squares: %.2f" % np.mean((y_hat - y) ** 2))


##

# Polynomial regression

print('\n\nPolynomial regression\n')

cdf = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]
train = cdf[msk]
test = cdf[~msk]


train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])


poly = PolynomialFeatures(degree=2)
train_x_poly = poly.fit_transform(train_x)
print(train_x_poly)

clf = linear_model.LinearRegression()
train_y_ = clf.fit(train_x_poly, train_y)
# The coefficients
print('Coefficients: ', clf.coef_)
print('Intercept: ', clf.intercept_)


axs[1][0].scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
XX = np.arange(0.0, 10.0, 0.1)
yy = clf.intercept_[0] + clf.coef_[0][1]*XX + clf.coef_[0][2]*np.power(XX, 2)
axs[1][0].plot(XX, yy, '-r')
axs[1][0].set_title('Polynomial regression')
axs[1][0].set_xlabel("Engine size")
axs[1][0].set_ylabel("Emission")


plt.show(block=False)

test_x_poly = poly.fit_transform(test_x)
test_y_ = clf.predict(test_x_poly)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y_ , test_y) )


# Practice Polynomial regression

print('\n\nPolynomial regression (cubic)\n')

poly3 = PolynomialFeatures(degree=3)
train_x_poly3 = poly3.fit_transform(train_x)
clf3 = linear_model.LinearRegression()
train_y_3 = clf3.fit(train_x_poly3, train_y)
# The coefficients
print('Coefficients: ', clf3.coef_)
print('Intercept: ', clf3.intercept_)

# Evaluation
test_x_poly3 = poly3.fit_transform(test_x)
test_y_3 = clf3.predict(test_x_poly3)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_3 - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_3 - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y_3 , test_y) )

# Plot
axs[1][1].scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
XX = np.arange(0.0, 10.0, 0.1)
yy = clf3.intercept_[0] + clf3.coef_[0][1]*XX + clf3.coef_[0][2]*np.power(XX, 2) + clf3.coef_[0][3]*np.power(XX, 3)
axs[1][1].plot(XX, yy, '-r')
axs[1][1].set_title('Polynomial regression (cubic)')
axs[1][1].set_xlabel("Engine size")
axs[1][1].set_ylabel("Emission")

plt.show()
print()
