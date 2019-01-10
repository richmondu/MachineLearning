import numpy # Scientific Computing library
import pandas # Python Data Analysis library
import matplotlib.pyplot as plot # Python 2D Plotting library
from sklearn import linear_model # Python Machine Learning library
from sklearn.metrics import mean_squared_error, r2_score # Python Machine Learning library
from sklearn.preprocessing import PolynomialFeatures



def plot_data(title, xlabel, ylabel, datalegend, linelegend, X, y, XX, z):
	plot.title(title)
	plot.xlabel(xlabel)
	plot.ylabel(ylabel)
	plot.scatter(X, y, color="green", marker="o", edgecolors='k', label=datalegend)
	plot.plot(XX, z, color="yellow", label=linelegend)
	plot.legend()
	plot.show()
	
def predict(x):
	test = numpy.c_[x]
	result = regression.predict(test)
	print('Predict({0}) = {1}'.format(test, result))
	return result

	
# read data using pandas
dataframe = pandas.read_csv('1_polynomial_regression_data.csv')
x_values = dataframe[['Change']]
y_values = dataframe[['Flowing']]

# separate data into training and testing data sets
x_values_train = x_values[:]
y_values_train = y_values[:]

# train model on training data set using scikit-learn
poly = PolynomialFeatures(degree=8)
x_values_train_poly = poly.fit_transform(x_values_train['Change'].values.reshape(-1,1))
regression = linear_model.LinearRegression()
regression.fit(x_values_train_poly, y_values_train)

# visualize training, testing and precition data sets using matplotlib
plot_data(
	"Polynomial Regression: Degree 8",
	"Change in water level (x)",
	"Water flowing out of the dam (y)",
	"Training Data",
	"Best Fit Line",
	x_values_train.values,
	y_values_train.values,
	numpy.linspace(-60,45),
	regression.intercept_+ numpy.sum(regression.coef_*poly.fit_transform(numpy.linspace(-60,45).reshape(-1,1)), axis=1)
	)
