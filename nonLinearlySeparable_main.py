from matplotlib import pyplot as plt
import numpy as np
from PerceptronMain import Perceptron
from PerceptronMain import plot_dataset
from PerceptronMain import load_data
from PerceptronMain import plot_one_vs_all
from PerceptronMain import plot_result
from PerceptronMain import final_accuracy

# def plot_final_result(X, Y, title, plot_no):
# 	plt.figure(plot_no, figsize=(8,5))
# 	plt.title(title)
# 	colors = ['b', 'g', 'r' ]

# 	first_class = np.array([])
# 	second_class = np.array([])
# 	third_class = np.array([])
# 	for index in range(Y.shape[0]):
# 		if Y[index] == 0:
# 			first_class = np.append(first_class, X[index])
# 		elif Y[index] == 1:
# 			second_class = np.append(second_class, X[index])

# 	first_class = first_class.reshape((int (first_class.shape[0] / 2), 2))
# 	second_class = second_class.reshape((int (second_class.shape[0] / 2), 2))
# 	# third_class = third_class.reshape((int (third_class.shape[0] / 2), 2))
		
# 	x = [point[0] for point in first_class]
# 	y = [point[1] for point in first_class]
# 	plt.scatter(x,y, c=colors[0],label="class 0")

# 	x = [point[0] for point in second_class]
# 	y = [point[1] for point in second_class]
# 	plt.scatter(x,y, c=colors[1],label="class 1")

# 	# x = [point[0] for point in third_class]
# 	# y = [point[1] for point in third_class]
# 	# plt.scatter(x,y, c=colors[2],label="class 2")

# 	plt.legend(loc=0)


def nonLinearlySeparable():
	plot_no = 0
	class_count = 2
	dirname = "nonLinearlySeparable/"


	# Load and show training data
	separated_train = load_data("train", class_count, dirname)
	# pprint(separated_train)
	plot_dataset(separated_train, "Training data", plot_no)
	plot_no += 1

	separated_test = load_data("test", class_count, dirname)
	plot_dataset(separated_test, "Testing data", plot_no)
	plot_no += 1

	positive_class = 0
	negative_class = 1

	train_set = list() # list of rows
	actual_y = list()
	for y, classdata in enumerate(separated_train):
		for datapoint in classdata:
			train_set.append(datapoint)
			if y == 0:
				actual_y.append(1)
			else:
				actual_y.append(-1)

	number_train = len(actual_y)
	X_train = np.array(train_set)
	y_train = np.array(actual_y)
	print(f"The number of training samples is {number_train}")
	print(f"The shape of X_train is {X_train.shape}")
	print(f"The shape of y_train is {y_train.shape}")

	test_set = list() # list of rows
	predicted_y = list()
	for y, classdata in enumerate(separated_test):
		for datapoint in classdata:
			test_set.append(datapoint)
			if y == positive_class:
				predicted_y.append(1)
			else:
				predicted_y.append(-1)

	number_test = len(predicted_y)
	X_test = np.array(test_set)
	y_test = np.array(predicted_y)
	print(f"The number of training samples is {number_test}")
	print(f"The shape of X_train is {X_test.shape}")
	print(f"The shape of y_train is {y_test.shape}")
	
	perceptron = Perceptron()
	perceptron.fit(X_train, y_train, 100)
	pred_y, accuracy, X_cross_weights = perceptron.score(X_test, y_test)

	plot_result(X_test, pred_y, "Result of model on Test Dataset", plot_no, "Class 1", "Class 2")
	plot_no += 1

	print(f"Accuracy of the model of Test Dataset is {accuracy * 100}%");
	plt.show()