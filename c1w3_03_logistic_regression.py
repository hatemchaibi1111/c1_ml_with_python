import os
import urllib.request
import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import jaccard_score
from sklearn.metrics import classification_report, confusion_matrix
import itertools
from sklearn.metrics import log_loss

filename = 'ChurnData.csv'
if not os.path.exists(filename):
    url = ('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/'
           'IBMDeveloperSkillsNetwork-ML0101EN-Coursera/labs/Data_files/ChurnData.csv')
    urllib.request.urlretrieve(url, filename)

churn_df = pd.read_csv("ChurnData.csv")
print('\nChurnData.csv\n' + str(churn_df.head()) + '\n...\n')

churn_df = churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip',   'callcard', 'wireless','churn']]
churn_df['churn'] = churn_df['churn'].astype('int')

X = np.asarray(churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']])
y = np.asarray(churn_df['churn'])


# normalize the dataset
X = preprocessing.StandardScaler().fit(X).transform(X)

# split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
print('Train set:', X_train.shape,  y_train.shape)
print('Test set:', X_test.shape,  y_test.shape)


# LogisticRegression from Scikit-learn package implements logistic regression and can use different
# numerical optimizers to find parameters, including ‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’ solvers.
#
# The version of Logistic Regression in Scikit-learn, support regularization. Regularization is a
# technique used to solve the overfitting problem in machine learning models.
# C parameter indicates inverse of regularization strength which must be a positive float.
# Smaller values specify stronger regularization.
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train, y_train)

# predict_proba returns estimates for all classes, ordered by the label of classes. So, the first column is
# the probability of class 1, P(Y=1|X), and second column is probability of class 0, P(Y=0|X)
yhat = LR.predict(X_test)
yhat_prob = LR.predict_proba(X_test)

print(jaccard_score(y_test, yhat))


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


print(confusion_matrix(y_test, yhat, labels=[1,0]))
# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, yhat, labels=[1,0])
np.set_printoptions(precision=2)


# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['churn=1','churn=0'],normalize= False,  title='Confusion matrix')
print(classification_report(y_test, yhat))

plt.show()
print()

print(log_loss(y_test, yhat_prob))


LR2 = LogisticRegression(C=0.05, solver='lbfgs').fit(X_train,y_train)
yhat2 = LR2.predict(X_test)
yhat_prob2 = LR2.predict_proba(X_test)

print(jaccard_score(y_test, yhat2))
print(confusion_matrix(y_test, yhat2, labels=[1,0]))
print(classification_report(y_test, yhat2))
print(log_loss(y_test, yhat_prob2))
