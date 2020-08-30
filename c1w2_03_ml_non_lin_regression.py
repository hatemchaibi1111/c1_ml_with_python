import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import urllib.request

##

filename = 'china_gdp.csv'
if not os.path.exists(filename):
    url = ('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/'
           'china_gdp.csv')
    urllib.request.urlretrieve(url, filename)

df = pd.read_csv("china_gdp.csv")
print('\nchina_gdp.csv:\n' + str(df.head(10)) + '\n...\n')


fig, axs = plt.subplots(2, 2)

x_data, y_data = (df["Year"].values, df["Value"].values)
axs[0][0].plot(x_data, y_data, 'ro')
axs[0][0].set_ylabel('GDP')
axs[0][0].set_xlabel('Year')
plt.show(block=False)

X = np.arange(-5.0, 5.0, 0.1)
Y = 1.0 / (1.0 + np.exp(-X))

axs[0][1].plot(X, Y)
axs[0][1].set_ylabel('Dependent Variable')
axs[0][1].set_xlabel('Indepdendent Variable')
plt.show(block=False)


def sigmoid(x, Beta_1, Beta_2):
    y = 1 / (1 + np.exp(-Beta_1 * (x - Beta_2)))
    return y


beta_1 = 0.10
beta_2 = 1990.0

# logistic function
Y_pred = sigmoid(x_data, beta_1, beta_2)

# plot initial prediction against datapoints
axs[1][0].plot(x_data, Y_pred*15000000000000.)
axs[1][0].plot(x_data, y_data, 'ro')

plt.show(block=False)

# normalization
xdata =x_data/max(x_data)
ydata =y_data/max(y_data)

from scipy.optimize import curve_fit
popt, pcov = curve_fit(sigmoid, xdata, ydata)

# print the final parameters
print("beta_1 = %f, \nbeta_2 = %f" % (popt[0], popt[1]))

x = np.linspace(1960, 2015, 55)
x = x/max(x)

y = sigmoid(x, *popt)
axs[1][1].plot(xdata, ydata, 'ro', label='data')
axs[1][1].plot(x,y, linewidth=3.0, label='fit')
axs[1][1].legend(loc='best')
axs[1][1].set_ylabel('GDP')
axs[1][1].set_xlabel('Year')
plt.show(block=False)

# split data into train/test
msk = np.random.rand(len(df)) < 0.8
train_x = xdata[msk]
test_x = xdata[~msk]
train_y = ydata[msk]
test_y = ydata[~msk]

# build the model using train set
popt, pcov = curve_fit(sigmoid, train_x, train_y)

# predict using test set
y_hat = sigmoid(test_x, *popt)

# evaluation
print("Mean absolute error: %.2f" % np.mean(np.absolute(y_hat - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((y_hat - test_y) ** 2))
from sklearn.metrics import r2_score
print("R2-score: {:.2f}".format(r2_score(y_hat , test_y)))

plt.show()
print()
