import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, RobustScaler
import sklearn.metrics as sm

pd.set_option('display.max_columns', 7)
pd.set_option('display.width', None)

data = pd.read_csv('fuel_consumption.csv')
print(data.head(5))

# analiza
print(data.info())

print(data.describe())
print(data.describe(include=[object]))

# data clensing
data.ENGINESIZE.fillna(data.ENGINESIZE.mean(), inplace=True)
data.FUELTYPE.fillna(data.FUELTYPE.mode()[0], inplace=True)
data.TRANSMISSION.fillna(data.TRANSMISSION.mode()[0], inplace=True)

print(data.info())

# the year is 2014 in whole dataset,there is no correlation with that atribute
data = data.drop(columns='MODELYEAR')

sb.heatmap(data.corr(), square=True, center=0, annot=True)
plt.show()

cont_atributes = ['CYLINDERS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY', 'FUELCONSUMPTION_COMB',
                  'FUELCONSUMPTION_COMB_MPG', 'ENGINESIZE']
out = 'CO2EMISSIONS'

fig, axis = plt.subplots(3, 2, figsize=(10, 8))
i = 0
j = 0
for atr in cont_atributes:
    axis[i, j].scatter(data[atr], data[out], s=5)
    axis[i, j].set_xlabel(atr, fontsize=5)
    axis[i, j].set_ylabel(out, fontsize=5)
    j += 1
    if j % 2 == 0:
        i += 1
        j = 0

fig.tight_layout(pad=5)

plt.figure()
sb.barplot(x='FUELTYPE', y=out, data=data)
plt.show()

sb.barplot(x=out, y="MAKE", data=data)
plt.show()
# check
# print(data.groupby(['MAKE'])['CO2EMISSIONS'].mean())

sb.barplot(x=out, y='TRANSMISSION', data=data)
plt.show()

# sb.barplot(x=out, y = 'MODEL', data=data)
# plt.show()  ???? model is not a category?

sb.barplot(x=out, y='VEHICLECLASS', data=data)
plt.show()

# FUELCONSUMPTION_COMB_MPG is not linear so we do not take him
cont_atributes.remove('FUELCONSUMPTION_COMB_MPG')
input_var = ['VEHICLECLASS', 'ENGINESIZE', 'CYLINDERS', 'TRANSMISSION', 'FUELTYPE', 'FUELCONSUMPTION_CITY',
             'FUELCONSUMPTION_HWY', 'FUELCONSUMPTION_COMB']

# make i model izbacujemo jer zasto bi bila bitna marka?
data.drop(columns=['MAKE', 'MODEL', 'FUELCONSUMPTION_COMB_MPG'], inplace=True)

# fueltype, transmission i vehicleclass cemo da onehot encodujemo
ohe = OneHotEncoder(dtype=int, sparse_output=False)
# fit_transform zahteva promenu oblika
fueltype = ohe.fit_transform(data['FUELTYPE'].to_numpy().reshape(-1, 1))
data.drop(columns=['FUELTYPE'], inplace=True)
data = data.join(pd.DataFrame(data=fueltype, columns=ohe.get_feature_names_out(['FUELTYPE'])))

transmission = ohe.fit_transform(data['TRANSMISSION'].to_numpy().reshape(-1, 1))
data.drop(columns=['TRANSMISSION'], inplace=True)
data = data.join(pd.DataFrame(data=transmission, columns=ohe.get_feature_names_out(['TRANSMISSION'])))

vehicleclass = ohe.fit_transform(data['VEHICLECLASS'].to_numpy().reshape(-1, 1))
data.drop(columns=['VEHICLECLASS'], inplace=True)
data = data.join(pd.DataFrame(data=vehicleclass, columns=ohe.get_feature_names_out(['VEHICLECLASS'])))

# normalizacija numerickih podataka
robust_scaler = RobustScaler()
data[cont_atributes] = robust_scaler.fit_transform(data[cont_atributes])

y = data[['CO2EMISSIONS']]
X = data.drop(columns=out)

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=234, shuffle=True)

# gotov model
print('\n')
print('\n')
print('Rezultati gotovog modela: ')
linear_model = LinearRegression().fit(X_train, y_train)

print("Koeficijenti: ")
print(linear_model.coef_)
print(linear_model.intercept_)
print('\n')
print("Trening set: ")
print("R2 score: %.10f" % linear_model.score(X_train, y_train))
print("Root Mean squared error: %.10f" % np.sqrt(mean_squared_error(y_train, linear_model.predict(X_train))))
print('\n')
print("Test set: ")
print("R2 score: %.10f" % linear_model.score(X_test, y_test))
print("Root Mean squared error: %.10f" % np.sqrt(mean_squared_error(y_test, linear_model.predict(X_test))))
print('\n')
print('\n')


# moj model
class LinearRegressionModel:
    def __init__(self, lr=0.01, iter=10000):
        self.lr = lr
        self.iter = iter
        self.cost = []
        self.coef = []

    def fit(self, x, y):
        self.coef = np.zeros((x.shape[1], 1))
        num = x.shape[0]

        for it in range(self.iter):
            error = np.dot(x, self.coef) - y
            grad = np.dot(x.T, error)
            self.coef -= (self.lr / num) * grad
            c = np.sum((error ** 2)) / (2 * num)
            self.cost.append(c)
        return self

    def predict(self, x):
        return np.dot(x, self.coef)


model = LinearRegressionModel()

model = model.fit(X_train, y_train)

predicted = model.predict(X_train)

print('Rezultati pravljenog modela: ')
print("Trening set: ")
print("R2 score =", round(sm.r2_score(y_train, predicted), 10))
print("Root Mean squared error =", round(np.sqrt(sm.mean_squared_error(y_train, predicted)), 10))
print('\n')

print("Test set: ")
test_pred = model.predict(X_test)
print("R2 score =", round(sm.r2_score(y_test, test_pred), 10))
print("Root Mean squared error =", round(np.sqrt(sm.mean_squared_error(y_test, test_pred)), 10))
