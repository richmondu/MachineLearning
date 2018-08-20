import numpy # Scientific Computing library
import pandas # Python Data Analysis library
import matplotlib.pyplot as plot # Python 2D Plotting library
from sklearn import linear_model # Python Machine Learning library
from sklearn.metrics import mean_squared_error, r2_score # Python Machine Learning library



def plot_data():
	plot.title("Linear Regression: Profits of Different Cities")
	plot.xlabel("Population of City in 10,000s")
	plot.ylabel("Profit in $10,000s")
	plot.scatter(x_values_train, y_values_train, color="green", marker="o", edgecolors='k', label='Training Data')
	plot.scatter(x_values_validation, y_values_validation, color="red", marker="x", label='Validation Data')
	plot.scatter(x_values_test, y_values_test, color="blue", marker="+", label='Testing Data')
	plot.plot(x_values_train, regression.predict(x_values_train), color="yellow", label='Best Fit Line')
	plot.legend()
	plot.show()
	
def predict(x):
	test = numpy.c_[x]
	result = regression.predict(test)
	print('Predict({0}) = {1}'.format(test, result))
	return result

	
# read data using pandas
dataframe = pandas.read_csv('1_linear_regression_data.csv')
x_values = dataframe[['Population']]
y_values = dataframe[['Profit']]

# separate data into training and testing data sets
x_values_train = x_values[0:60]
y_values_train = y_values[0:60]
x_values_validation = x_values[60:80]
y_values_validation = y_values[60:80]
x_values_test = x_values[80:100]
y_values_test = y_values[80:100]

# train model on training data set using scikit-learn
regression = linear_model.LinearRegression()
regression.fit(x_values_train, y_values_train)

# test model on testing data set
y_values_predict = regression.predict(x_values_validation)
print('\nMetrics')
print('Coefficients: ', regression.coef_)
print('Mean squared error: ', mean_squared_error(y_values_validation, y_values_predict))
print('Variance score: ', r2_score(y_values_validation, y_values_predict)) # 1 is perfect prediction)

# visualize training, testing and precition data sets using matplotlib
plot_data()

# predict some values
predict(5)
predict(15)
predict(25)
