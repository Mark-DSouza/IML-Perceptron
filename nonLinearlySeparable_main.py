from matplotlib import pyplot as plt
import numpy as np
from PerceptronMain import Perceptron
from PerceptronMain import plot_dataset
from PerceptronMain import load_data
from PerceptronMain import plot_one_vs_all
from PerceptronMain import plot_result
from PerceptronMain import final_accuracy

def nonLinearlySeparable(dirname):
	plot_no = 0
	class_count = 2
	# dirname = "nonLinearlySeparable/"


	# Load and show training data
	separated_train = load_data("train", class_count, dirname)
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