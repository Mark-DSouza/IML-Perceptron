from matplotlib import pyplot as plt
import numpy as np
from PerceptronModel import Perceptron
from PerceptronMain import plot_dataset
from PerceptronMain import load_data
from PerceptronMain import plot_one_vs_all
from PerceptronMain import plot_result
from PerceptronMain import final_accuracy

def plot_final_result(X, Y, title, plot_no):
	plt.figure(plot_no, figsize=(8,5))
	plt.title(title)
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


def linearlySeparable():
	plot_no = 0
	class_count = 3
	dirname = "linearlySeparable/"
	# dirname = "overlapping/"


	# Load and show training data
	separated_train = load_data("train", class_count, dirname)
	# pprint(separated_train)
	plot_dataset(separated_train, "Training data", plot_no)
	plot_no += 1

	separated_test = load_data("test", class_count, dirname)
	plot_dataset(separated_test, "Testing data", plot_no)
	plot_no += 1

	number_train = 0
	number_test = 0
	list_y_test = list()

	for current_class in range(class_count):
		plot_one_vs_all(separated_dataset=separated_train, title="train", current_class=current_class, plot_no=plot_no)
		plot_no += 1


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
		
		perceptron = Perceptron()
		perceptron.fit(X_train, y_train, 100)
		pred_y, accuracy, X_cross_weights = perceptron.score(X_test, y_test)

		y_final = np.concatenate((y_final, X_cross_weights.reshape((number_test, 1))), axis=1)

		plot_result(X_test, pred_y, "Result of test prediction", plot_no)
		plot_no += 1


	y_final = np.delete(y_final, 0, 1) #not important
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
	plot_final_result(X_test, result, 'Test Result for 3 classes', plot_no)
	plot_no += 1


	list_y_test = np.array(list_y_test)
	list_y_test = list_y_test.reshape((list_y_test.shape[0], 1))
	print(final_accuracy(list_y_test, result)) # To find accuracy
	plt.show()
