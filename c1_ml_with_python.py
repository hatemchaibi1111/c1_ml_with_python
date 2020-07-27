import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import pylab as pl


import urllib.request

filename = 'FuelConsumption.csv'

if not os.path.exists(filename):
    url = 'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/FuelConsumptionCo2.csv'
    urllib.request.urlretrieve(url, filename)

df = pd.read_csv(filename)

df_described = df.describe()
print(df_described)

cdf = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]

##
#%% code cell

viz = cdf[['CYLINDERS', 'ENGINESIZE', 'CO2EMISSIONS', 'FUELCONSUMPTION_COMB']]
viz.hist()
plt.show()

print(df.head())
