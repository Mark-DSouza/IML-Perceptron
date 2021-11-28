from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from PerceptronMain import Perceptron
from PerceptronMain import plot_dataset
from PerceptronMain import load_data
from PerceptronMain import plot_result

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

def calculate_confusion_matrix(y_test, y_pred):
	y_pred = np.vectorize(lambda x: 1 if x==1 else 2)(y_pred)
	y_test = np.vectorize(lambda x: 1 if x==1 else 2)(y_test)
	actual_ones = np.vectorize(lambda val: 1 if val == 1 else 0)(y_test)
	elementwiseWithOne = np.multiply(actual_ones, y_pred)
	confusion1comma1 = np.mean(elementwiseWithOne == 1)
	confusion1comma2 = np.mean(elementwiseWithOne == 2)

	actual_twos = np.vectorize(lambda val: 1 if val == 2 else 0)(y_test)
	elementwiseWithTwo = np.multiply(actual_twos, y_pred)
	confusion2comma1 = np.mean(elementwiseWithTwo == 1)
	confusion2comma2 = np.mean(elementwiseWithTwo == 2)

	return [
		[confusion1comma1, confusion1comma2],
		[confusion2comma1, confusion2comma2],
	]

def display_metrics(y_test, y_pred, class_count=2):
	conf_mat = calculate_confusion_matrix(y_test, y_pred)
	print()
	print("Confusion Matrix:")
	print("-----------------")
	print(f"                Predicted Class 1 | Predicted Class 2")
	for i in range(class_count):
		print(f"Actual class {i+1}:", end="")
		for j in range(class_count):
			print(f"{conf_mat[i][j]*100:17.4f} %", end="")
		print()
	print()
	prec_sum = 0
	rec_sum = 0
	f1_sum = 0
	for class_label in range(class_count):
		correctly_predicted_as_positive = conf_mat[class_label][class_label]
		total_predicted_as_positive = sum([conf_mat[i][class_label] for i in range(class_count)])
		total_actual_positive = sum([conf_mat[class_label][i] for i in range(class_count)])

		prec = correctly_predicted_as_positive/total_predicted_as_positive
		rec = correctly_predicted_as_positive/total_actual_positive

		f1_score = 2*prec*rec/(prec+rec)
		prec_sum += prec
		rec_sum += rec
		f1_sum += f1_score
		
		print(f"Class {class_label+1}: ", end="")
		print(f"Precision = {prec:8.4f}, Recall = {rec:8.4f}, F1-score = {f1_score:8.4f}")

	print()
	print(f"Mean precision is {prec_sum/class_count:.4f}")
	print(f"Mean recall is {rec_sum/class_count:.4f}")
	print(f"Mean F1-score is {f1_sum/class_count:.4f}")
	print()

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
	
	perceptron = Perceptron()
	perceptron.fit(X_train, y_train, 100)
	pred_y, accuracy, _, _ = perceptron.score(X_test, y_test)

	plot_result(X_test, pred_y, "Result of model on Test Dataset", plot_no, "Class 1", "Class 2")
	plot_no += 1

	print(f"Accuracy of the model of Test Dataset is {accuracy * 100}%");
	display_metrics(y_test, pred_y)

	plt.figure(plot_no, figsize=(8,5))
	plot_no += 1
	plt.title(f"class 1 VS class 2 Decision Region on Training Dataset")
	plot_decision_boundary(perceptron, X_train, y_train, colormap=ListedColormap(['b', 'g', 'r']))
	
	
	plt.figure(plot_no, figsize=(8,5))
	plot_no += 1
	plt.title(f"class 1 VS class 2 Decision Region on Test Dataset")
	plot_decision_boundary(perceptron, X_test, y_test, colormap=ListedColormap(['b', 'g', 'r']))
	plt.show()