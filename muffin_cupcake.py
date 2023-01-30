from sklearn.metrics import mean_squared_error
from random import randint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
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

# cupcake = 0, muffin = 1
type = LabelEncoder()

label = type.fit_transform(data['type'])
data.drop(columns='type', inplace=True)
data['type'] = label

X = data.drop(columns='type')
y = data[['type']]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=234, shuffle=True)

scaler = MinMaxScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

y_test = y_test.values
y_train = y_train.values


def round_up_to_odd(f):
    f = int(np.ceil(f))
    return f + 1 if f % 2 == 0 else f


k = round_up_to_odd(np.sqrt(len(X_train)))

classifier = KNeighborsClassifier(n_neighbors=13, metric='manhattan')
classifier.fit(X_train, y_train)

print('\n')
print("Gotov model: ")
y_pred_test = classifier.predict(X_test)
print("Test acc: %.5f" % classifier.score(X_test, y_test))

mse = mean_squared_error(y_test, y_pred_test)
print(f'mse: {mse}')
print('\n')


class KNNClassifier:
    def __init__(self, k, x_train, y_train):
        self.k = k
        self.x_train = x_train
        self.y_train = y_train

    def manhattan_distance(self, point1, point2):
        return sum(abs(value1 - value2) for value1, value2 in zip(point1, point2))

    def euclidian(self, data1, data2):
        return np.linalg.norm(data1 - data2)

    def predict(self, x_test):
        y_test = []

        for data in x_test:
            distance = []

            for train_data in self.x_train:
                distance.append(self.manhattan_distance(train_data, data))

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
knnModel = KNNClassifier(13, X_train, y_train)
test_predict = np.array(knnModel.predict(X_test))

print("Test acc: %.5f" % accuracy_score(y_test, test_predict.reshape(len(test_predict), 1)))
mse = mean_squared_error(y_test, test_predict)
print(f'mse: {mse}')

"""
Zapazanja:
Kada koristim MinMaxScaler imam bolje rezultat nego kada koristim StandardScaler.
Takodje se pokazuje da je menhetn distanca bolja od euklidove
Najbolje je kada je k = 13
"""

accuracies = []
ks = range(1, 40)

for k in ks:
    knn = KNNClassifier(k, X_train, y_train)
    predicted = np.array(knn.predict(X_test))
    accuracy = accuracy_score(y_test, predicted.reshape(len(predicted), 1))
    accuracies.append(accuracy)

fig, axis = plt.subplots()
axis.plot(ks, accuracies)
axis.set(xlabel="k", ylabel="Accuracy", title="Performanse knn-a u zavisnosti od k")
plt.show()
