import numpy as np
import pandas as pd
import os
import urllib.request
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

filename = 'drug200.csv'
if not os.path.exists(filename):
    url = ('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/'
           'drug200.csv')
    urllib.request.urlretrieve(url, filename)

my_data = pd.read_csv("drug200.csv", delimiter=",")


X = my_data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values

# Sklearn Decision Trees do not handle categorical variables. But still we can convert these features to numerical
# values.pandas.get_dummies() Convert categorical variable into dummy / indicator variables.

le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F', 'M'])
X[:, 1] = le_sex.transform(X[:, 1])

le_BP = preprocessing.LabelEncoder()
le_BP.fit(['LOW', 'NORMAL', 'HIGH'])
X[:, 2] = le_BP.transform(X[:, 2])

le_Chol = preprocessing.LabelEncoder()
le_Chol.fit(['NORMAL', 'HIGH'])
X[:, 3] = le_Chol.transform(X[:, 3])

y = my_data["Drug"]

X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)

drugTree = DecisionTreeClassifier(criterion="entropy", max_depth=4)
drugTree.fit(X_trainset,y_trainset)

predTree = drugTree.predict(X_testset)

print (predTree [0:5])
print (y_testset [0:5])

from sklearn import metrics
import matplotlib.pyplot as plt
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_testset, predTree))


def calculate_jaccard_index(array1, array2):
    length_overlap = len(array1[(array1 == array2)])
    return length_overlap/(len(array1) + len(array2) - length_overlap)


print("DecisionTrees's jaccard_index (calculated): ", calculate_jaccard_index(y_testset.values, predTree))


def accuracy(array1, array2):
    length_overlap = len(array1[(array1 == array2)])
    return length_overlap/len(array1)

print("DecisionTrees's Accuracy (calculated): ", accuracy(y_testset.values, predTree))

from sklearn.externals.six import StringIO
import pydotplus
import matplotlib.image as mpimg
from sklearn import tree

dot_data = StringIO()
filename = "drugtree.png"
featureNames = my_data.columns[0:5]
targetNames = my_data["Drug"].unique().tolist()
out=tree.export_graphviz(drugTree, feature_names=featureNames, out_file=dot_data,
                         class_names=np.unique(y_trainset), filled=True,  special_characters=True, rotate=False)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png(filename)
img = mpimg.imread(filename)
plt.figure(figsize=(100, 200))
plt.imshow(img, interpolation='nearest')
print()


