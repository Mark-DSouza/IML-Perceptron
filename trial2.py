import numpy as np

y_pred = np.array([
    [1],
    [1],
    [1],
    [2],
    [2],
    [3],
    [3],
    [3],
])

y_test = np.array([
    [1],
    [1],
    [1],
    [2],
    [2],
    [2],
    [3],
    [3],
])

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

# print(f"1 comma 2, {np.mean(y_test)}")

# print(y_pred == y_test)
# print(np.mean(y_pred == 1))
# print(np.mean(y_pred == 2))
# print(np.mean(y_pred == 3))