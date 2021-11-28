from matplotlib import pyplot as plt
from class_vs_class import class_vs_class
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
	plt.scatter(x,y, c=colors[0],label="class 1")

	x = [point[0] for point in second_class]
	y = [point[1] for point in second_class]
	plt.scatter(x,y, c=colors[1],label="class 2")

	x = [point[0] for point in third_class]
	y = [point[1] for point in third_class]
	plt.scatter(x,y, c=colors[2],label="class 3")

	plt.legend(loc=0)

def calculate_confusion_matrix(y_test, y_pred):
	y_pred = np.vectorize(lambda x: 1 if x==0 else (2 if x==1 else 3))(y_pred)
	y_test = np.vectorize(lambda x: 1 if x==0 else (2 if x==1 else 3))(y_test)
	actual_ones = np.vectorize(lambda val: 1 if val == 1 else 0)(y_test)
	elementwiseWithOne = np.multiply(actual_ones, y_pred)
	confusion1comma1 = np.mean(elementwiseWithOne == 1)
	confusion1comma2 = np.mean(elementwiseWithOne == 2)
	confusion1comma3 = np.mean(elementwiseWithOne == 3)
	print(confusion1comma1, confusion1comma2, confusion1comma3)

	actual_twos = np.vectorize(lambda val: 1 if val == 2 else 0)(y_test)
	elementwiseWithTwo = np.multiply(actual_twos, y_pred)
	confusion2comma1 = np.mean(elementwiseWithTwo == 1)
	confusion2comma2 = np.mean(elementwiseWithTwo == 2)
	confusion2comma3 = np.mean(elementwiseWithTwo == 3)
	print(confusion2comma1, confusion2comma2, confusion2comma3)

	actual_threes = np.vectorize(lambda val: 1 if val == 3 else 0)(y_test)
	elementwiseWithThree = np.multiply(actual_threes, y_pred)
	confusion3comma1 = np.mean(elementwiseWithThree == 1)
	confusion3comma2 = np.mean(elementwiseWithThree == 2)
	confusion3comma3 = np.mean(elementwiseWithThree == 3)
	print(confusion3comma1, confusion3comma2, confusion3comma3)

	return [
		[confusion1comma1, confusion1comma2, confusion1comma3],
		[confusion2comma1, confusion2comma2, confusion2comma3],
		[confusion3comma1, confusion3comma2, confusion3comma3],
	]

def display_metrics(y_test, y_pred, class_count=3):
	conf_mat = calculate_confusion_matrix(y_test, y_pred)
	print()
	print("Confusion Matrix:")
	print("-----------------")
	print(f"                Predicted Class 1 | Predicted Class 2 | Predicted Class 3")
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

def linearlySeparable(dirname):
	plot_no = 0
	class_count = 3
	# dirname = "linearlySeparable/"
	# dirname = "overlapping/"


	# Load and show training data
	separated_train = load_data("train", class_count, dirname)
	plot_dataset(separated_train, "Training data", plot_no)
	plot_no += 1

	separated_test = load_data("test", class_count, dirname)
	plot_dataset(separated_test, "Testing data", plot_no)
	plot_no += 1

	number_train = 0
	number_test = 0
	list_y_test = list()

	for current_class in range(class_count):
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
		X_test = np.array(test_set)
		y_test = np.array(predicted_y)
		
		perceptron = Perceptron()
		perceptron.fit(X_train, y_train, 100)
		pred_y, accuracy, X_cross_weights = perceptron.score(X_test, y_test)

		y_final = np.concatenate((y_final, X_cross_weights.reshape((number_test, 1))), axis=1)

		# plot_result(X_test, pred_y, "Result of test prediction", plot_no)
		# plot_no += 1


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


	list_y_test = np.array(list_y_test) # to computer overall accuracy
	list_y_test = list_y_test.reshape((list_y_test.shape[0], 1))
	print(f"The Overall accuracy of the multi-class model is {final_accuracy(list_y_test, result) * 100}%")

	display_metrics(list_y_test, result)

	for i in range(class_count):
		for j in range(i+1, class_count):
			class_vs_class(separated_train=separated_train, separated_test=separated_test, class1=i, class2=j, plot_no=plot_no)
			plot_no+=3
	# class_vs_class(separated_train=separated_train, separated_test=separated_test, class1=0, class2=1, plot_no=plot_no)
	# class_vs_class(separated_train=separated_train, separated_test=separated_test, class1=0, class2=2, plot_no=plot_no)
	# class_vs_class(separated_train=separated_train, separated_test=separated_test, class1=1, class2=2, plot_no=plot_no)
	# plot_no+=2

	plt.show()
