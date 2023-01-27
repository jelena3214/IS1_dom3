import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 7)
pd.set_option('display.width', None)

data = pd.read_csv('fuel_consumption.csv')
print(data.head(5))

# analiza
print(data.info())

print(data.describe())
print(data.describe(include=[object]))

# data clensing
data.ENGINESIZE.fillna(data.ENGINESIZE.mean, inplace=True)
data.FUELTYPE.fillna(data.FUELTYPE.mode()[0], inplace=True)
data.TRANSMISSION.fillna(data.TRANSMISSION.mode()[0], inplace=True)

print(data.info())

# the year is 2014 in whole dataset,there is no correlation with that atribute
data.drop(columns=['MODELYEAR'], inplace=True)

plt.figure()
# in case we have atribute with no correlation he will be shown in lightblue
color = plt.get_cmap('RdYlGn')  # default color
color.set_bad('lightblue')
sb.heatmap(data.corr(), cmap=color, square=True, annot=True)
plt.show()

cont_atributes = ['CYLINDERS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY', 'FUELCONSUMPTION_COMB',
                  'FUELCONSUMPTION_COMB_MPG']
out = 'CO2EMISSIONS'

# TODO add labels and axis
fig = plt.figure()
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

fig.delaxes(axis[2, 1])
fig.tight_layout(pad=5)
plt.show()

plt.figure()
sb.barplot(x='FUELTYPE', y=out, data=data)
plt.show()

#TODO: popravi prikaz svi ostali su ovog tipa prikaza
plt.figure(figsize=(10,10))
plt.legend(loc="upper left")
data.groupby('MAKE').size().plot(kind='pie', legend = True, labeldistance=None, autopct='%1.1f%%')
plt.show()
