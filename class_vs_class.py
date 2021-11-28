from matplotlib import pyplot as plt
from numpy.core import numeric
from utils import load_class_data
from planar_utils import load_planar_dataset, plot_decision_boundary
import sklearn
import numpy as np
from PerceptronModel import Perceptron
from PerceptronMain import plot_result

def plot_class_vs_class(separated_dataset, title, class1, class2):
	plt.figure(100, figsize=(8,5))
	plt.title(title)
	colors = ['b', 'g', 'r' ]
	
	for class_ind in {class1, class2}:
		x = [point[0] for point in separated_dataset[class_ind]]
		y = [point[1] for point in separated_dataset[class_ind]]
		plt.scatter(x,y, c=colors[class_ind],label=f"class {class_ind}")
	plt.legend(loc=0)

def class_vs_class(separated_train, separated_test, class1, class2):
	plot_class_vs_class(separated_train, f"Class {class1} VS Class {class2} Train", class1, class2)
	plot_class_vs_class(separated_test, f"Class {class1} VS Class {class2} Test", class1, class2)
	train_set = list() # list of rows
	actual_y = list()
	for y, classdata in enumerate(separated_train):
		for datapoint in classdata:
			train_set.append(datapoint)
			if y == class1:
				actual_y.append(1)
			elif y == class2:
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
			if y == class1:
				predicted_y.append(1)
			elif y == class2:
				predicted_y.append(-1)

	number_test = len(predicted_y)
	print(number_test)
	X_test = np.array(test_set)
	y_test = np.array(predicted_y)
	
	perceptron = Perceptron()
	perceptron.fit(X_train, y_train, 100)
	final_y, accuracy, actual_predictions = perceptron.score(X_test, y_test)

	print(f"The accuray of class {class1+1} VS class {class2+1} is {accuracy}")
	plot_result(X_test, final_y, "Result of test prediction")
	

	
