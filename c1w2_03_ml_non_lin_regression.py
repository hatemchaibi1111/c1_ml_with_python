import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import urllib.request

##

filename = 'china_gdp.csv'
if not os.path.exists(filename):
    url = ('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/'
           'china_gdp.csv')
    urllib.request.urlretrieve(url, filename)

df = pd.read_csv("china_gdp.csv")
print(df.head(10))

plt.figure(figsize=(8,5))
x_data, y_data = (df["Year"].values, df["Value"].values)
plt.plot(x_data, y_data, 'ro')
plt.ylabel('GDP')
plt.xlabel('Year')
plt.show()

print()
