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


def plot_final_result(X, Y, title):
	global plot_no
	plt.figure(plot_no, figsize=(8,5))
	plt.title(title)
	plot_no += 1
	colors = ['b', 'g', 'r' ]

	first_class = np.array([])
	second_class = np.array([])
	third_class = np.array([])
	for index in range(Y.shape[0]):
		if Y[index] == 0:
			first_class = np.append(first_class, X[index])
		elif Y[index] == 1:
			second_class = np.append(second_class, X[index])
		else:
			third_class = np.append(third_class, X[index])

	first_class = first_class.reshape((int (first_class.shape[0] / 2), 2))
	second_class = second_class.reshape((int (second_class.shape[0] / 2), 2))
	third_class = third_class.reshape((int (third_class.shape[0] / 2), 2))
		
	x = [point[0] for point in first_class]
	y = [point[1] for point in first_class]
	plt.scatter(x,y, c=colors[0],label="class 0")

	x = [point[0] for point in second_class]
	y = [point[1] for point in second_class]
	plt.scatter(x,y, c=colors[1],label="class 1")

	x = [point[0] for point in third_class]
	y = [point[1] for point in third_class]
	plt.scatter(x,y, c=colors[2],label="class 2")

	plt.legend(loc=0)



	# for class_ind in range(len(X)):
	# 	x0 = [point[0] for point in X[class_ind]]
	# 	x1 = [point[1] for point in X[class_ind]]
	# 	x0_mean = mean(x0)
	# 	x1_mean = mean(x1)
	# 	plt.scatter(X, y, c=colors, cmap=matplotlib.colors.ListedColormap(colors))
	# for x0, x1 in X:

def final_accuracy(y_test, result):
	return np.mean(y_test == result)

seed(1)

class_count = 3
dirname = "linearlySeparable/"
# current_class = 0

# Load and show training data
separated_train = load_data("train", class_count, dirname)
# pprint(separated_train)
plot_dataset(separated_train, title="Training data")

separated_test = load_data("test", class_count, dirname)
plot_dataset(separated_test, title="Testing data")

number_train = 0
number_test = 0
list_y_test = list()
for current_class in range(class_count):
	plot_one_vs_all(separated_dataset=separated_train, title="train", current_class=current_class)

	train_set = list() # list of rows
	actual_y = list()
	for y, classdata in enumerate(separated_train):
		for datapoint in classdata:
			train_set.append(datapoint)
			if y == current_class:
				actual_y.append(1)
			else:
				actual_y.append(-1)

	number_train = len(actual_y)
	print(number_train)
	X_train = np.array(train_set)
	y_train = np.array(actual_y)

	

	test_set = list() # list of rows
	predicted_y = list()
	for y, classdata in enumerate(separated_test):
		for datapoint in classdata:
			test_set.append(datapoint)
			if current_class == 0:
				list_y_test.append(y)

			if y == current_class:
				predicted_y.append(1)
			else:
				predicted_y.append(-1)

	number_test = len(predicted_y)
	if current_class == 0:
		y_final  = np.empty((number_test, 1))
	print(number_test)
	X_test = np.array(test_set)
	y_test = np.array(predicted_y)
	


	# plt.show()
	perceptron = Perceptron()
	perceptron.fit(X_train, y_train, 100)
	final_y, accuracy, actual_predictions = perceptron.score(X_test, y_test)

	# pprint(final_y)
	# print(final_y.shape)
	# print(accuracy)
	# print(actual_predictions)

	y_final = np.concatenate((y_final, actual_predictions.reshape((number_test, 1))), axis=1)

	plot_result(X_test, final_y, "Result of test prediction")
	# plt.show()

y_final = np.delete(y_final, 0, 1)
result = list()
for index in range(y_final.shape[0]):
	if (y_final[index][0] > y_final[index][1]) and (y_final[index][0] > y_final[index][2]):
		result.append(0)
	elif (y_final[index][1] > y_final[index][2]) and (y_final[index][1] > y_final[index][0]):
		result.append(1)
	else:
		result.append(2)

result = np.array(result)
result = result.reshape((result.shape[0], 1))
plot_final_result(X_test, result, 'FINALLY')
plt.show()

list_y_test = np.array(list_y_test)
list_y_test = list_y_test.reshape((list_y_test.shape[0], 1))
# print(list_y_test)
# print("How this is the next data")
# print(result)
print(final_accuracy(list_y_test, result))