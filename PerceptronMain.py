# Naive Bayes On The Iris Dataset
from matplotlib import pyplot as plt
# from csv import reader
from random import seed
from random import randrange
from math import sqrt
from math import exp
from math import pi
from pprint import pprint

from numpy.core import numeric
from utils import load_class_data
from planar_utils import load_planar_dataset, plot_decision_boundary
import sklearn
import numpy as np
from PerceptronModel import Perceptron

plot_no = 0

# Return the data as: list(list(list():datapoint):classdata):dataset
def load_data(type, classcount, dirname):
	return [ 
		load_class_data(dirname+filename) for filename in [
			f"class{i+1}_{type}.txt" for i in range(classcount)
		] 
	]

# Plot of training data with mean displayed in different color
def plot_dataset(separated_dataset, title):
	global plot_no
	colors = ['b', 'g', 'r' ]
	mean_colors = ['c', 'm', 'y']
	# separated = separate_by_class(dataset)
	
	plt.figure(plot_no, figsize=(8,5))
	plt.title(title)
	plot_no += 1
	

	for class_ind in range(len(separated_dataset)):
		x = [point[0] for point in separated_dataset[class_ind]]
		y = [point[1] for point in separated_dataset[class_ind]]
		x_mean = mean(x)
		y_mean = mean(y)
		plt.scatter(x,y, c=colors[class_ind],label="class "+str(class_ind+1))
		plt.scatter(x_mean,y_mean, c=mean_colors[class_ind],label="class "+str(class_ind+1)+" mean")
		plt.legend(loc=0)

	# plt.show()

# Calculate the mean of a list of numbers
def mean(numbers):
	# return np.sum(numbers)/float(len(numbers))
	return np.mean(numbers)

def plot_one_vs_all(separated_dataset, title, current_class):
	global plot_no
	plt.figure(plot_no, figsize=(8,5))
	plt.title(title)
	plot_no += 1
	colors = ['b', 'g' ]
	mean_colors = ['c', 'm']
	
	for class_ind in range(len(separated_dataset)):
		x = [point[0] for point in separated_dataset[class_ind]]
		y = [point[1] for point in separated_dataset[class_ind]]
		x_mean = mean(x)
		y_mean = mean(y)
		if class_ind == current_class:
			plt.scatter(x,y, c=colors[0],label="class relavant")
			# plt.scatter(x_mean,y_mean, c=mean_colors[class_ind],label="class "+str(class_ind+1)+" mean")
		else: 
			plt.scatter(x,y, c=colors[1],label="class other")
			# plt.scatter(x_mean,y_mean, c=mean_colors[class_ind],label="class "+str(class_ind+1)+" mean")
		plt.legend(loc=0)

def plot_result(X, Y, title):
	global plot_no
	plt.figure(plot_no, figsize=(8,5))
	plt.title(title)
	plot_no += 1
	colors = ['b', 'g' ]
	mean_colors = ['c', 'm']

	positive_class = np.array([])
	negative_class = np.array([])

	for index in range(Y.shape[0]):
		if Y[index] == 1:
			positive_class = np.append(positive_class, X[index])
		else:
			negative_class = np.append(negative_class, X[index])
			
	positive_class = positive_class.reshape((int (positive_class.shape[0] / 2), 2))
	negative_class = negative_class.reshape((int (negative_class.shape[0] / 2), 2))
	print(positive_class.shape, negative_class.shape)
	# print(positive_class, negative_class)

	x = [point[0] for point in positive_class]
	y = [point[1] for point in positive_class]
	plt.scatter(x,y, c=colors[0],label="class positive")

	x = [point[0] for point in negative_class]
	y = [point[1] for point in negative_class]
	plt.scatter(x,y, c=colors[1],label="class negative")
	
	plt.legend(loc=0)





	# for class_ind in range(len(X)):
	# 	x0 = [point[0] for point in X[class_ind]]
	# 	x1 = [point[1] for point in X[class_ind]]
	# 	x0_mean = mean(x0)
	# 	x1_mean = mean(x1)
	# 	plt.scatter(X, y, c=colors, cmap=matplotlib.colors.ListedColormap(colors))
	# for x0, x1 in X:

seed(1)

class_count = 3
dirname = "linearlySeparable/"
current_class = 0

# Load and show training data
separated_train = load_data("train", class_count, dirname)
# pprint(separated_train)
plot_dataset(separated_train, title="Training data")
plot_one_vs_all(separated_dataset=separated_train, title="train", current_class=current_class)

# train_set = list() # list of rows
# for y, classdata in enumerate(separated_train):
# 	for datapoint in classdata:
# 		train_set.append(datapoint + [y])

train_set = list() # list of rows
actual_y = list()
for y, classdata in enumerate(separated_train):
	for datapoint in classdata:
		train_set.append(datapoint)
		if y == 0:
			actual_y.append(1)
		else:
			actual_y.append(-1)


# pprint(train_set)
# pprint(actual_y)

X_train = np.array(train_set)
y_train = np.array(actual_y)
# y_train = y_train.reshape(y_train.shape[0], 1)

# print(X.shape, y.shape)

# Load and show test data
separated_test = load_data("test", class_count, dirname)
# pprint(separated_test)
plot_dataset(separated_test, title="Testing data")

test_set = list() # list of rows
predicted_y = list()
for y, classdata in enumerate(separated_test):
	for datapoint in classdata:
		test_set.append(datapoint)
		if y == 0:
			predicted_y.append(1)
		else:
			predicted_y.append(-1)

X_test = np.array(test_set)
y_test = np.array(predicted_y)
# y_test = y_test.reshape(y_test.shape[0], 1)

# print(X_test.shape, y_test.shape)


# plt.show()
perceptron = Perceptron()
perceptron.fit(X_train, y_train, 100)
final_y, accuracy, class_weights = perceptron.score(X_test, y_test)

# pprint(final_y)
print(final_y.shape)
print(accuracy)
print(class_weights)

plot_result(X_test, final_y, "Result of test prediction")
plt.show()