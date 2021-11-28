from matplotlib import pyplot as plt
import numpy as np
from PerceptronMain import Perceptron
from PerceptronMain import plot_dataset
from PerceptronMain import load_data
from PerceptronMain import plot_one_vs_all
from PerceptronMain import plot_result
from PerceptronMain import final_accuracy

def calculate_confusion_matrix(y_test, y_pred):
	y_pred = np.vectorize(lambda x: 1 if x==1 else 2)(y_pred)
	y_test = np.vectorize(lambda x: 1 if x==1 else 2)(y_test)
	actual_ones = np.vectorize(lambda val: 1 if val == 1 else 0)(y_test)
	elementwiseWithOne = np.multiply(actual_ones, y_pred)
	confusion1comma1 = np.mean(elementwiseWithOne == 1)
	confusion1comma2 = np.mean(elementwiseWithOne == 2)
	print(confusion1comma1, confusion1comma2)

	actual_twos = np.vectorize(lambda val: 1 if val == 2 else 0)(y_test)
	elementwiseWithTwo = np.multiply(actual_twos, y_pred)
	confusion2comma1 = np.mean(elementwiseWithTwo == 1)
	confusion2comma2 = np.mean(elementwiseWithTwo == 2)
	print(confusion2comma1, confusion2comma2)

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
	display_metrics(y_test, pred_y)
	plt.show()