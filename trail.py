from typing import Coroutine

from matplotlib.pyplot import axis
import numpy as np

# arr = np.array([[1, 2, 3], [4, 5, 6], [4, 5, 7], [4, 5, 8]])
# y = np.array([1, -1, 1, 1])
# correct = np.array([])
# incorrect = np.array([])

# for index in range(y.shape[0]):
#     if y[index] == 1:
#         correct = np.append(correct, arr[index])
#     else:
#         incorrect = np.append(incorrect, arr[index])
        
# correct = correct.reshape((int (correct.shape[0] / 3), 3))
# incorrect = incorrect.reshape((int (incorrect.shape[0] / 3), 3))
# print(correct.shape, incorrect.shape)
# print(correct, incorrect)

ans = np.empty((5, 1))
print(ans)
for i in range(5):
    curr = np.ones((5, 1))
    ans = np.concatenate((ans, curr), axis=1)
ans = np.delete(ans, 0, 1)
print(ans)