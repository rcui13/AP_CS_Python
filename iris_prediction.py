import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.cluster import MeanShift

# from random import *
import random
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

import numpy as np
from sklearn.cluster import KMeans


# set column names
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
# read csv file, set names as column names
dataset = pd.read_csv('iris.csv', names=names)

dataset.replace("Iris-setosa", 0, inplace=True)
dataset.replace("Iris-versicolor", 1, inplace=True)
dataset.replace("Iris-virginica", 2, inplace=True)

original_df = pd.DataFrame.copy(dataset)


print(dataset.head())

# print row x column size
# print(dataset.shape)

# print first 'n'
# print(dataset.head(20))

# print last 'n'
# print(dataset.tail(20))

# print count, mean, std, 5 number summary
# print(dataset.describe())

# print size of 'class'
# print(dataset.groupby('class').size())


# turn dataset into array
array = dataset.values
# data values of each row
X = array[:, 0:4]
# classification of data
y = array[:, 4]
# X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.20, random_state=random.randint(0, 10))
# print(X[0:5])

# make predictions on validation dataset
# model = KNeighborsClassifier()
# model = SVC(gamma='auto')
# model = LogisticRegression(solver='liblinear', multi_class='ovr')
# model.fit(X_train, Y_train)
# predictions = model.predict(X_test)
#
# print(accuracy_score(Y_test, predictions))
# print(confusion_matrix(Y_test, predictions))
# print(classification_report(Y_test, predictions))

clf = MeanShift()
clf.fit(X)
# centroids = clf.cluster_centers_
labels = clf.labels_
cluster_centers = clf.cluster_centers_
original_df['cluster_group'] = np.nan
for i in range(len(X)):
    original_df['cluster_group'].iloc[i] = labels[i]

n_clusters_ = len(np.unique(labels))
flower_rates = {}
for i in range(n_clusters_):
    temp_df = original_df[ (original_df['cluster_group'] == float(i))]
    flower_cluster = temp_df[(temp_df['class'] == 1)]
    flower_rate = len(flower_cluster)/len(temp_df)
    flower_rates[i] = flower_rate

print(flower_rates)
print(original_df[(original_df['cluster_group']==0)])
colors = ["g.", "y.", "c."]

correct = 0
for i in range(len(X)):
    plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize=25)
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = clf.predict(predict_me)
    print(prediction)

    if prediction[0] == y[i]:
        correct += 1

print(correct)
print(len(X))
print(correct/len(X))

plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=150, linewidths=5)
plt.show()
