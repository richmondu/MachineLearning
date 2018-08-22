import sys, os
import numpy as np # Scientific Computing library
import pandas # Python Data Analysis library
import matplotlib.pyplot as plot # Python 2D Plotting library
import matplotlib.patches as patch # Python 2D Plotting library
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn import svm # Python Machine Learning library
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope



def plot_decision_boundary(X, y, clf):
  
	ax = plot.gca()
	xlim = ax.get_xlim()
	ylim = ax.get_ylim()
	# print(xlim)
	# print(ylim)

	# Create grid to evaluate model
	xx = np.linspace(xlim[0], xlim[1])
	yy = np.linspace(ylim[0], ylim[1])
	# print(xx)
	# print(yy)  
	YY, XX = np.meshgrid(yy, xx)
	# print(YY)
	# print(XX)   
	xy = np.vstack([XX.ravel(), YY.ravel()]).T
	# print(xy)    
	Z = clf.decision_function(xy).reshape(XX.shape)
	# print(Z)  

	# ax.contour(XX, YY, Z, colors='yellow', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
	ax.contour(XX, YY, Z, colors='yellow', levels=[0], alpha=0.5, linestyles=['-'])


def plot_data(title, xlabel, ylabel, oklegend, nglegend, X, y, clf):

	plot.figure()
	plot.title(title)
	plot.xlabel(xlabel)
	plot.ylabel(ylabel)
	
	plot.scatter(X[:, 0], X[:, 1], c = np.where(y == 1,'g','r'), edgecolors='black')
	
	plot_decision_boundary(X, y, clf)
	
	legend_ok = patch.Patch(label=oklegend, color='green', edgecolor='black')
	legend_ng = patch.Patch(label=nglegend, color='red', edgecolor='black')
	legend_boundary = patch.Patch(label='Decision Boundary', color='yellow', edgecolor='black')
	plot.legend(handles=[legend_ok, legend_ng, legend_boundary])
	plot.show()


# read data and filter using pandas
dataframe = pandas.read_csv('4_isolation_forest_data.csv')
x_values = dataframe[['Latency', 'Throughput']]

x_values_train = x_values[:]
#x_values_test = x_values[:]

# Create a IsolationForest OutlierDetection/AnomalyDetection classifier
# Note: can use OneClassSVM, EllipticEnvelope or IsolationForest
#clf = svm.OneClassSVM(nu=0.03, kernel="rbf", gamma=0.1)
#clf = EllipticEnvelope(contamination=0.019)
clf = IsolationForest(contamination=0.03) #contamination=0.017)
clf.fit(dataframe)
pred = clf.predict(dataframe)
dataframe['Prediction'] = pred
# print(dataframe['Prediction'].values)

# Plot the data and decision boundary
plot_data(
	"Anomaly Detection using Isolation Forest", 
	"Latency (ms)", 
	"Throughput (mbps)",
	"Normal",
	"Anomalous",
	x_values_train.values, 
	dataframe['Prediction'].values, 
	clf)

	