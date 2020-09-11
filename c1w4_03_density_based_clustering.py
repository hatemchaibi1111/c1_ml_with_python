import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
import os
import urllib.request

import pandas as pd


def createDataPoints(centroidLocation, numSamples, clusterDeviation):
    # Create random data and store in feature matrix X and response vector y.
    X, y = make_blobs(n_samples=numSamples, centers=centroidLocation,
                      cluster_std=clusterDeviation)

    # Standardize features by removing the mean and scaling to unit variance
    X = StandardScaler().fit_transform(X)
    return X, y


X, y = createDataPoints([[4, 3], [2, -1], [-1, 4]], 1500, 0.5)

# Modeling
# DBSCAN stands for Density-Based Spatial Clustering of Applications with Noise. This technique is one of the most
# common clustering algorithms which works based on density of object. The whole idea is that if a particular point
# belongs to a cluster, it should be near to lots of other points in that cluster.
#
# It works based on two parameters: Epsilon and Minimum Points
# Epsilon determine a specified radius that if includes enough number of points within, we call it dense area
# minimumSamples determine the minimum number of data points we want in a neighborhood to define a cluster.

epsilon = 0.3
minimumSamples = 7
db = DBSCAN(eps=epsilon, min_samples=minimumSamples).fit(X)
labels = db.labels_

# Distinguish outliers
# Lets Replace all elements with 'True' in core_samples_mask that are in the cluster,
# 'False' if the points are outliers.

core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
unique_labels = set(labels)

colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

# Plot the points with colors
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = 'k'

    class_member_mask = (labels == k)

    # Plot the datapoints that are clustered
    xy = X[class_member_mask & core_samples_mask]
    plt.scatter(xy[:, 0], xy[:, 1],s=50, c=[col], marker=u'o', alpha=0.5)

    # Plot the outliers
    xy = X[class_member_mask & ~core_samples_mask]
    plt.scatter(xy[:, 0], xy[:, 1],s=50, c=[col], marker=u'o', alpha=0.5)

plt.show(block=False)

# To better underestand differences between partitional and density-based clusteitng,
# try to cluster the above dataset into 3 clusters using k-Means.
k_means = KMeans(init="k-means++", n_clusters=3, n_init=12)
k_means.fit(X)
k_means_labels = k_means.labels_
k_means_cluster_centers = k_means.cluster_centers_
# Initialize the plot with the specified dimensions.
fig = plt.figure(figsize=(6, 4))
colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means_labels))))
ax = fig.add_subplot(1, 1, 1)
for k, col in zip(range(len([[4, 4], [-2, -1], [2, -3], [1, 1]])), colors):
    my_members = (k_means_labels == k)
    cluster_center = k_means_cluster_centers[k]
    ax.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.')
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=6)
ax.set_title('KMeans')
ax.set_xticks(())
ax.set_yticks(())
plt.show(block=False)

# Weather Station Clustering using DBSCAN & scikit-learn


filename = 'weather-stations20140101-20141231.csv'
if not os.path.exists(filename):
    url = ('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/'
           'IBMDeveloperSkillsNetwork-ML0101EN-Coursera/labs/Data_files/weather-stations20140101-20141231.csv')
    urllib.request.urlretrieve(url, filename)

pdf = pd.read_csv(filename)
pdf = pdf[pd.notnull(pdf["Tm"])]
pdf = pdf.reset_index(drop=True)


# 6 Visualization of clusters based on location

import os
os.environ['PROJ_LIB'] = r'C:\ProgramData\Anaconda3\pkgs\proj4-5.2.0-ha925a31_1\Library\share'
# os.environ['PROJ_LIB'] = r'C:/ProgramData/Anaconda3/envs/pricingenv/lib/site-packages/mpl_toolkits/basemap'
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from pylab import rcParams

rcParams['figure.figsize'] = (14, 10)

llon = -140
ulon = -50
llat = 40
ulat = 65

pdf = pdf[(pdf['Long'] > llon) & (pdf['Long'] < ulon) & (pdf['Lat'] > llat) &(pdf['Lat'] < ulat)]
fig = plt.figure(3)
my_map = Basemap(projection='merc',
                 resolution='l', area_thresh=1000.0,
                 llcrnrlon=llon, llcrnrlat=llat,  # min longitude (llcrnrlon) and latitude (llcrnrlat)
                 urcrnrlon=ulon, urcrnrlat=ulat)  # max longitude (urcrnrlon) and latitude (urcrnrlat)

my_map.drawcoastlines()
my_map.drawcountries()
# my_map.drawmapboundary()
my_map.fillcontinents(color='white', alpha=0.3)
my_map.shadedrelief()

# To collect data based on stations

xs, ys = my_map(np.asarray(pdf.Long), np.asarray(pdf.Lat))
pdf['xm'] = xs.tolist()
pdf['ym'] = ys.tolist()

# Visualization1
for index, row in pdf.iterrows():
    # x,y = my_map(row.Long, row.Lat)
    my_map.plot(row.xm, row.ym,markerfacecolor=([1,0,0]),  marker='o', markersize= 5, alpha = 0.75)
# plt.text(x,y,stn)
plt.show(block=False)


# 7- Clustering of stations based on their location, mean, max, and min Temperature

import sklearn.utils
from sklearn.preprocessing import StandardScaler
sklearn.utils.check_random_state(1000)
Clus_dataSet = pdf[['xm', 'ym']]
Clus_dataSet = np.nan_to_num(Clus_dataSet)
Clus_dataSet = StandardScaler().fit_transform(Clus_dataSet)

# Compute DBSCAN
db = DBSCAN(eps=0.15, min_samples=10).fit(Clus_dataSet)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
pdf["Clus_Db"] = labels

realClusterNum=len(set(labels)) - (1 if -1 in labels else 0)
clusterNum = len(set(labels))

# 8- Visualization of clusters based on location and Temperture
fig = plt.figure(4)
rcParams['figure.figsize'] = (14, 10)

my_map = Basemap(projection='merc',
                 resolution='l', area_thresh=1000.0,
                 llcrnrlon=llon, llcrnrlat=llat,  # min longitude (llcrnrlon) and latitude (llcrnrlat)
                 urcrnrlon=ulon, urcrnrlat=ulat)  # max longitude (urcrnrlon) and latitude (urcrnrlat)

my_map.drawcoastlines()
my_map.drawcountries()
# my_map.drawmapboundary()
my_map.fillcontinents(color='white', alpha=0.3)
my_map.shadedrelief()

# To create a color map
colors = plt.get_cmap('jet')(np.linspace(0.0, 1.0, clusterNum))
# Visualization
for clust_number in set(labels):
    c = (([0.4, 0.4, 0.4]) if clust_number == -1 else colors[np.int(clust_number)])
    clust_set = pdf[pdf.Clus_Db == clust_number]
    my_map.scatter(clust_set.xm, clust_set.ym, color=c,  marker='o', s=20, alpha=0.85)
    if clust_number != -1:
        cenx = np.mean(clust_set.xm)
        ceny = np.mean(clust_set.ym)
        plt.text(cenx, ceny, str(clust_number), fontsize=25, color='red',)
        print("Cluster "+str(clust_number)+', Avg Temp: ' + str(np.mean(clust_set.Tm)))

plt.show(block=False)
print()

