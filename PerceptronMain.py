# Perceptron on Linearly Separable
import numpy as np

from matplotlib import pyplot as plt
from utils import load_class_data
from PerceptronModel import Perceptron


# Return the data as: list(list(list():datapoint):classdata):dataset
def load_data(type, classcount, dirname):
	return [ 
		load_class_data(dirname+filename) for filename in [
			f"class{i+1}_{type}.txt" for i in range(classcount)
		] 
	]


# Plot of training data with mean displayed in different color
def plot_dataset(separated_dataset, title, plot_no):
	colors = ['b', 'g', 'r' ]
	mean_colors = ['c', 'm', 'y']
	# separated = separate_by_class(dataset)
	
	plt.figure(plot_no, figsize=(8,5))
	plt.title(title)
	

	for class_ind in range(len(separated_dataset)):
		x = [point[0] for point in separated_dataset[class_ind]]
		y = [point[1] for point in separated_dataset[class_ind]]
		x_mean = mean(x)
		y_mean = mean(y)
		plt.scatter(x,y, c=colors[class_ind],label="class "+str(class_ind+1))
		plt.scatter(x_mean,y_mean, c=mean_colors[class_ind],label="class "+str(class_ind+1)+" mean")
		plt.legend(loc=0)


# Calculate the mean of a list of numbers
def mean(numbers):
	return np.mean(numbers)

def plot_one_vs_all(separated_dataset, title, current_class, plot_no):
	plt.figure(plot_no, figsize=(8,5))
	plt.title(title)
	colors = ['b', 'g' ]
	
	for class_ind in range(len(separated_dataset)):
		x = [point[0] for point in separated_dataset[class_ind]]
		y = [point[1] for point in separated_dataset[class_ind]]
		if class_ind == current_class:
			plt.scatter(x,y, c=colors[0],label="class relavant")
		else: 
			plt.scatter(x,y, c=colors[1],label="class other")
		plt.legend(loc=0)

def plot_result(X, Y, title, plot_no, positive_class_label="positive class", negative_class_label="negative class"):
	plt.figure(plot_no, figsize=(8,5))
	plt.title(title)
	colors = ['b', 'g' ]

	positive_class = np.array([])
	negative_class = np.array([])

	for index in range(Y.shape[0]):
		if Y[index] == 1:
			positive_class = np.append(positive_class, X[index])
		else:
			negative_class = np.append(negative_class, X[index])
			
	positive_class = positive_class.reshape((int (positive_class.shape[0] / 2), 2))
	negative_class = negative_class.reshape((int (negative_class.shape[0] / 2), 2))

	x = [point[0] for point in positive_class]
	y = [point[1] for point in positive_class]
	plt.scatter(x,y, c=colors[0],label=positive_class_label)

	x = [point[0] for point in negative_class]
	y = [point[1] for point in negative_class]
	plt.scatter(x,y, c=colors[1],label=negative_class_label)
	
	plt.legend(loc=0)


def final_accuracy(y_test, result):
	return np.mean(y_test == result)

def plot_decision_boundary(perceptron, X, y, colormap):
	X = X.T
	# Set min and max values and give it some padding
	x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
	y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
	h_x = (x_max-x_min)/100
	h_y = (y_max-y_min)/100
	# Generate a grid of points with distance h between them
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h_x), np.arange(y_min, y_max, h_y))
	# Predict the function value for the whole grid
	final_y, _, _, _ = perceptron.score(np.c_[xx.ravel(), yy.ravel()], y)
	Z = final_y
	Z = Z.reshape(xx.shape)
	# Plot the contour and training examples
	plt.contourf(xx, yy, Z, alpha=0.25, cmap=colormap)
	plt.ylabel('x2')
	plt.xlabel('x1')
	plt.scatter(X[0, :], X[1, :], c=y, cmap=colormap)
