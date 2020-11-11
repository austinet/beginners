from activkitTF import *
import linear_regression as lr
import numpy as np


# x_data = np.array([[1., 2.],
#                    [2., 3.],
#                    [3., 1.],
#                    [4., 3.],
#                    [5., 3.],
#                    [6., 2.]], dtype=np.float32)
# y_data = np.array([[0.],
#                    [0.],
#                    [0.],
#                    [1.],
#                    [1.],
#                    [1.]], dtype=np.float32)

# logistic = logistic(x_data, y_data)
# logistic.regression(epoch=3000, learning_rate=0.01)


x_data = np.array([[1, 2, 1, 1],
                   [2, 1, 3, 2],
                   [3, 1, 3, 4],
                   [4, 1, 5, 5],
                   [1, 7, 5, 5],
                   [1, 2, 5, 6],
                   [1, 6, 6, 6],
                   [1, 7, 7, 7]], dtype=np.float32)
y_data = np.array([[0, 0, 1],
                   [0, 0, 1],
                   [0, 0, 1],
                   [0, 1, 0],
                   [0, 1, 0],
                   [0, 1, 0],
                   [1, 0, 0],
                   [1, 0, 0]], dtype=np.float32)

softmax = softmax(x_data, y_data)
softmax.regression(epoch=10000, learning_rate=0.01)