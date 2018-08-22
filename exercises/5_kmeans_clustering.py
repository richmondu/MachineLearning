import sys, os
import numpy as np # Scientific Computing library
import pandas # Python Data Analysis library
import matplotlib.pyplot as plot # Python 2D Plotting library
from sklearn.cluster import KMeans



def plot_data(title, xlabel, ylabel, oklegend, nglegend, X, clf):

	fig = plot.figure()
	plot.title(title)
	plot.xlabel(xlabel)
	plot.ylabel(ylabel)
	
	plot.scatter(X[:,0], X[:,1], c=clf.labels_, cmap=plot.cm.prism, label='data') 
	plot.scatter(clf.cluster_centers_[:,0], clf.cluster_centers_[:,1], marker='+', c='black', label='centroids');

	plot.legend()
	plot.show()


# read data and filter using pandas
dataframe = pandas.read_csv('5_kmeans_clustering.csv')
x_values = dataframe[['x', 'y']]
x_values_train = x_values[:]
#x_values_test = x_values[:]

# apply K-means clustering
clf = KMeans(n_clusters=3)
clf.fit(x_values_train)

# plot the data and the clusters
plot_data(
	"K-Means Clustering", 
	"x value",
	"y value",
	"",
	"",
	x_values_train.values, 
	clf)
	