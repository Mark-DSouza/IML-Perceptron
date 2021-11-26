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


seed(1) 

class_count = 3
dirname = "overlapping/"

# Load and show training data
separated_train = load_data("train", class_count, dirname)
plot_dataset(separated_train, title="Training data")

train_set = list() # list of rows
for y, classdata in enumerate(separated_train):
	for datapoint in classdata:
		train_set.append(datapoint + [y])

# Load and show test data
separated_test = load_data("test", class_count, dirname)
plot_dataset(separated_test, title="Testing data")

test_set = list() # list of rows
for y, classdata in enumerate(separated_test):
	for datapoint in classdata:
		test_set.append(datapoint + [y])

plt.show()
