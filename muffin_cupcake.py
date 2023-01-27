from random import randint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv('cakes.csv')

print(data.head())
print(data.info())
print(data.describe())

sb.heatmap(data.corr(), square=True, center=0, annot=True)
plt.show()

col = data.columns.values.tolist()
col.remove('type')
out = 'type'

fig, axis = plt.subplots(3, 2, figsize=(10, 8))
i = 0
j = 0
for atr in col:
    sb.swarmplot(x=out, y=atr, data=data, ax=axis[i, j], s=3, palette="Set2", hue=out, legend=False)
    j += 1
    if j % 2 == 0:
        i += 1
        j = 0

plt.show()

scaler = MinMaxScaler()
data[col] = scaler.fit_transform(data[col])

# cupcake = 0, muffin = 1
type = LabelEncoder()

label = type.fit_transform(data['type'])
data.drop(columns='type', inplace=True)
data['type'] = label

X = data.drop(columns='type').values
y = data[['type']].values

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=234, shuffle=True)

classifier = KNeighborsClassifier(n_neighbors=round(np.sqrt(len(X))), metric='euclidean')
classifier.fit(X_train, y_train.ravel())

prediction = classifier.predict(X_train)

# TODO: jos neka metrika?
print('\n')
print("Gotov model: ")
print("Trening acc: %.5f" % accuracy_score(y_train, prediction))
y_pred_test = classifier.predict(X_test)
print("Test acc: %.5f" % accuracy_score(y_test, y_pred_test))
print('\n')
print('\n')


class KNNClassifierEuclidianDistance:
    def __init__(self, k, x_train, y_train):
        self.k = k
        self.neighbours = []
        self.x_train = x_train
        self.y_train = y_train

    def euclidian(self, data1, data2):
        return np.linalg.norm(data1 - data2)

    def predict(self, x_test):
        y_test = []

        for data in x_test:
            distance = []

            for train_data in self.x_train:
                distance.append(self.euclidian(train_data, data))

            distance = np.array(distance)
            near = np.argsort(distance)[:self.k]
            labels = self.y_train[near]

            zero_count = np.count_nonzero(labels == 0)
            one_count = np.count_nonzero(labels == 1)
            if zero_count > one_count:
                test_label = 0
            elif one_count > zero_count:
                test_label = 1
            else:
                test_label = randint(0, 1)

            y_test.append(test_label)
        return y_test


print("Rezultati napravljenog modela: ")
knnModel = KNNClassifierEuclidianDistance(round(np.sqrt(len(X))), X_train, y_train)
train_predict = knnModel.predict(X_train)
test_predict = knnModel.predict(X_test)

print("Trening acc: %.5f" % accuracy_score(y_train, train_predict))
print("Test acc: %.5f" % accuracy_score(y_test, test_predict))
