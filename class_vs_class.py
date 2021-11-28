from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from PerceptronModel import Perceptron
from PerceptronMain import plot_result
from PerceptronMain import plot_decision_boundary

def plot_class_vs_class(separated_dataset, title, class1, class2, plot_no):
	plt.figure(plot_no, figsize=(8,5))
	plt.title(title)
	colors = ['b', 'g', 'r' ]
	
	for class_ind in {class1, class2}:
		x = [point[0] for point in separated_dataset[class_ind]]
		y = [point[1] for point in separated_dataset[class_ind]]
		plt.scatter(x,y, c=colors[class_ind],label=f"class {class_ind+1}")
	plt.legend(loc=0)

def class_vs_class(separated_train, separated_test, class1, class2, plot_no):
	plot_class_vs_class(separated_train, f"Class {class1+1} VS Class {class2+1} Train", class1, class2, plot_no)
	plot_class_vs_class(separated_test, f"Class {class1+1} VS Class {class2+1} Test", class1, class2, plot_no+1)
	
	train_set = list() # list of rows
	actual_y = list()
	for y, classdata in enumerate(separated_train):
		for datapoint in classdata:
			if y == class1:
				train_set.append(datapoint)
				actual_y.append(1)
			elif y == class2:
				train_set.append(datapoint)
				actual_y.append(-1)

	number_train = len(actual_y)
	# print(number_train)
	X_train = np.array(train_set)
	y_train = np.array(actual_y)

	test_set = list() # list of rows
	predicted_y = list()
	for y, classdata in enumerate(separated_test):
		for datapoint in classdata:
			if y == class1:
				predicted_y.append(1)
				test_set.append(datapoint)
			elif y == class2:
				predicted_y.append(-1)
				test_set.append(datapoint)


	number_test = len(predicted_y)
	X_test = np.array(test_set)
	y_test = np.array(predicted_y)
	
	perceptron = Perceptron()
	perceptron.fit(X_train, y_train, 100)
	final_y, accuracy, actual_predictions, weights = perceptron.score(X_test, y_test)

	print(f"The accuray of class {class1+1} VS class {class2+1} is {accuracy*100}%")
	plot_result(X_test, final_y, f"class {class1+1} VS class {class2+1} Predictions on Test Dataset", plot_no+2, f"{class1+1}", f"{class2+1}")
	
	plt.figure(plot_no+3, figsize=(8,5))
	plt.title(f"class {class1+1} VS class {class2+1} Decision Region on Training Dataset")
	plot_decision_boundary(perceptron, X_train, y_train, colormap=ListedColormap(['r', 'g', 'b']))

	plt.figure(plot_no+4, figsize=(8,5))
	plt.title(f"class {class1+1} VS class {class2+1} Decision Region on Test Dataset")
	plot_decision_boundary(perceptron, X_test, y_test, colormap=ListedColormap(['r', 'g', 'b']))
