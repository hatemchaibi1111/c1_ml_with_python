import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

from sklearn import preprocessing
import os
import urllib.request

pd.set_option('display.width', 0)

filename = 'teleCust1000t.csv'
if not os.path.exists(filename):
    url = ('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/'
           'teleCust1000t.csv')
    urllib.request.urlretrieve(url, filename)

df = pd.read_csv("teleCust1000t.csv")

print('\nteleCust1000t.csv\n' + str(df.head()) + '\n...\n')

df.hist(column='income', bins=50)

plt.show(block=False)

##

X = df[['region', 'tenure', 'age', 'marital', 'address', 'income', 'ed', 'employ', 'retire', 'gender', 'reside']].values

# Normalize Data
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
y = df['custcat'].values

##
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
print('Train set:', X_train.shape,  y_train.shape)
print('Test set:', X_test.shape,  y_test.shape)

from sklearn.neighbors import KNeighborsClassifier
k = 4
# Train Model and             Predict
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
yhat = neigh.predict(X_test)


from sklearn import metrics
print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))

neigh2 = KNeighborsClassifier(n_neighbors=6).fit(X_train,y_train)
yhat2 = neigh.predict(X_test)
print("Train set Accuracy2: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy2: ", metrics.accuracy_score(y_test, yhat))


# accuracy of KNN for different Ks
Ks = 10
mean_acc = np.zeros((Ks - 1))
std_acc = np.zeros((Ks - 1))
ConfustionMx = [];
for n in range(1, Ks):
    # Train Model and Predict
    neigh = KNeighborsClassifier(n_neighbors=n).fit(X_train, y_train)
    yhat = neigh.predict(X_test)
    mean_acc[n - 1] = metrics.accuracy_score(y_test, yhat)

    std_acc[n - 1] = np.std(yhat == y_test) / np.sqrt(yhat.shape[0])


plt.figure(2)
plt.plot(range(1, Ks), mean_acc, 'g')
plt.fill_between(range(1, Ks), mean_acc - 1 * std_acc, mean_acc + 1 * std_acc, alpha=0.10)
plt.legend(('Accuracy ', '+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Nabors (K)')
plt.tight_layout()
plt.show(block=False)

print("The best accuracy was with", mean_acc.max(), "with k =", mean_acc.argmax()+1)

plt.show()
print()
