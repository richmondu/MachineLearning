import numpy # Scientific Computing library
import pandas # Python Data Analysis library
import matplotlib.pyplot as plot # Python 2D Plotting library
import matplotlib.patches as patch # Python 2D Plotting library
from sklearn import linear_model # Python Machine Learning library



def plot_data(with_decision_boundary):
	if with_decision_boundary == True:
		h = .02
		x_min, x_max = x_values_train['Exam 1'].min() - 5, x_values_train['Exam 1'].max() + 5
		y_min, y_max = x_values_train['Exam 2'].min() - 5, x_values_train['Exam 2'].max() + 5
		xx, yy = numpy.meshgrid(numpy.arange(x_min, x_max, h), numpy.arange(y_min, y_max, h))
		Z = regression.predict(numpy.c_[xx.ravel(), yy.ravel()])
		Z = Z.reshape(xx.shape)
		plot.figure(1)

		# decision boundary
		plot.contour(xx, yy, Z, c='yellow')
		# training data
		plot.scatter(x_values_train['Exam 1'], x_values_train['Exam 2'], c=numpy.where(dataframe['Final Exam'].values == 1,'g','r'), edgecolors='k')
		# testing data
		plot.scatter(x_values_test['Exam 1'], x_values_test['Exam 2'], c=numpy.where(dataframe_test['Final Exam'].values == 1,'g','r'), marker='x', edgecolors='k')

		plot.title("Logistic Regression: Final Exam Results based on Exam 1 and 2")
		plot.xlabel("Exam 1 score")
		plot.ylabel("Exam 2 score")
		legend_admitted = patch.Patch(label='Admitted', color='green')
		legend_notadmitted = patch.Patch(label='Not admitted', color='red')
		legend_decisionboundary = patch.Patch(label='Decision Boundary', color='yellow')
		plot.legend(handles=[legend_admitted, legend_notadmitted, legend_decisionboundary])
		plot.show()
	else:
		plot.title("Logistic Regression: Final Exam Results based on Exam 1 and 2")
		plot.xlabel("Exam 1 score")
		plot.ylabel("Exam 2 score")

		# training data
		plot.scatter(x_values_train[['Exam 1']], x_values_train[['Exam 2']], c=numpy.where(dataframe['Final Exam'].values == 1,'g','r'), marker="o", edgecolors="black")
		# testing data
		plot.scatter(x_values_test[['Exam 1']], x_values_test[['Exam 2']], c=numpy.where(dataframe_test['Final Exam'].values == 1,'g','r'), marker="x")

		legend_admitted = patch.Patch(label='Admitted', color='green', edgecolor='black')
		legend_notadmitted = patch.Patch(label='Not admitted', color='red', edgecolor='black')
		plot.legend(handles=[legend_admitted, legend_notadmitted])
		plot.show()

def predict(x, y):
	test = numpy.c_[x, y]
	result = regression.predict(test)
	print('Predict({0})={1} (confidence score={2})'.format(test, 'Admitted' if result==1 else 'Not admitted', regression.decision_function(test)))
	return result
	
	
# read data and filter using pandas
dataframe = pandas.read_csv('2_logistic_regression_data.csv')
x_values_train = dataframe[['Exam 1', 'Exam 2']]
y_values_train = dataframe[['Final Exam']]

# train model on training data set using scikit-learn
regression = linear_model.LogisticRegression(solver='newton-cg') # use newton-cg instead of default liblinear
regression.fit(x_values_train, y_values_train)

# test model on generated testing data set 
x_values_test = pandas.DataFrame(numpy.random.randint(low=20, high=100, size=(20, 2)), columns=['Exam 1', 'Exam 2'])
y_values_test_predict = regression.predict(x_values_test)
y_values_test = pandas.DataFrame(y_values_test_predict, columns=['Final Exam'])
dataframe_test = pandas.concat([x_values_test, y_values_test], axis=1)

# plot the training data, testing data and decision boundary using matplotlib
plot_data(True)

# predict some values
predict(40, 90)
predict(70, 60)
predict(90, 30)
