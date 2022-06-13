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
# names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
# read csv file, set names as column names
dataset = pd.read_csv('dataset.csv')

# dataset.replace("Iris-setosa", 0, inplace=True)
# dataset.replace("Iris-versicolor", 1, inplace=True)
# dataset.replace("Iris-virginica", 2, inplace=True)

original_df = pd.DataFrame.copy(dataset)


print(dataset.head())

# print row X_train column size
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
random.shuffle(array)
# data values of each row
X_train = array[:, 0:3]
# classification of data
Y_train = array[:, 3]
# X_train_train, X_train_test, Y_train, Y_test = train_test_split(X_train, y, test_size=0.20, random_state=1)
# print(X_train[0:5])

testset = pd.read_csv('testdata.csv')
arr = testset.values
X_test = arr[:,0:3]
y_test = arr[:,3]

# make predictions on validation dataset
# model = KNeighborsClassifier()
# model = SVC(gamma='auto')
# model = LogisticRegression(solver='liblinear', multi_class='ovr')
# model.fit(X_train_train, Y_train)
# predictions = model.predict(X_train_test)
#
# print(accuracy_score(y_test, predictions))
# print(confusion_matriX_train(y_test, predictions))
# print(classification_report(y_test, predictions))

clf = MeanShift()
clf.fit(X_train)
centroids = clf.cluster_centers_
labels = clf.labels_
cluster_centers = clf.cluster_centers_
original_df['cluster_group'] = np.nan
for i in range(len(X_train)):
    original_df['cluster_group'].iloc[i] = labels[i]

n_clusters_ = len(np.unique(labels))
flower_rates = {}
for i in range(n_clusters_):
    temp_df = original_df[ (original_df['cluster_group'] == float(i))]
    flower_cluster = temp_df[(temp_df['label'] == 1)]
    flower_rate = len(flower_cluster)/len(temp_df)
    flower_rates[i] = flower_rate

print(flower_rates)
print(original_df[(original_df['cluster_group']==0)])
colors = ["g.", "y.", "c."]

correct = 0
for i in range(len(X_train)):
    # plt.plot(X_train[i][0], X_train[i][1], colors[labels[i]], markersize=25)
    predict_me = np.array(X_train[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = clf.predict(predict_me)
    # print(prediction)

    if prediction[0] == Y_train[i]:
        correct += 1

print(correct)
print(len(X_train))
print(correct/len(X_train))

# plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=150, linewidths=5)
# plt.show()
