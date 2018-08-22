import sys, os
import numpy as np # Scientific Computing library
import pandas # Python Data Analysis library
import matplotlib.pyplot as plot # Python 2D Plotting library
import matplotlib.patches as patch # Python 2D Plotting library
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import svm # Python Machine Learning library



def plot_decision_function_helper(X, y, clf):
  
	plot.scatter(X[:, 0], X[:, 1], c = np.where(y == 1,'g','r'), edgecolors='black')
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
	
	plot_decision_function_helper(X, y, clf)
	
	legend_ok = patch.Patch(label=oklegend, color='green', edgecolor='black')
	legend_ng = patch.Patch(label=nglegend, color='red', edgecolor='black')
	legend_boundary = patch.Patch(label='Decision Boundary', color='yellow', edgecolor='black')
	plot.legend(handles=[legend_ok, legend_ng, legend_boundary])
	plot.show()


# read data and filter using pandas
dataframe = pandas.read_csv('3_support_vector_machine_data.csv')
x_values = dataframe[['Test 1', 'Test 2']]
y_values = dataframe[['Final Test']]
x_values_train = x_values[:100]
y_values_train = y_values[:100]
# x_values_test = x_values[100:]
# y_values_test = y_values[100:]

# Create a linear SVM classifier 
clf = svm.SVC(kernel='rbf', C = 1000000.0)
clf.fit(x_values_train, y_values_train)

# Plot the data and decision boundary
# print(x_values_train.values)
# print(np.transpose(y_values_train.values)[0])
plot_data(
	"Support Vector Machine: Final test based on Test 1 & 2", 
	"Test 1 score", 
	"Test 2 score", 
	"Passed",
	"Failed",
	x_values_train.values, 
	np.transpose(y_values_train.values)[0], 
	clf)

