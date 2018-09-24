import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy import linalg as LA
import math
from matplotlib.colors import ListedColormap
from BayesClassifier import BayesClassifier
from GlobalClassifier import GlobalClassifier

# data preprocessing for different datasets
"""
x = pd.read_csv('dataset2', sep=" ", header = None)
x1 = x.iloc[:500, :2].values
x2 = x.iloc[500:1000, :2].values
x3 = x.iloc[1000:, :2].values

y1 = np.full(500, 1) # class-1
y2 = np.full(500, 2) # class-2
y3 = np.full(1000, 3) # class-3

total_size = 1500

x_one = pd.read_csv('1Class1.txt', sep=" ", header = None)
x_two = pd.read_csv('1Class2.txt', sep=" ", header = None)
x_three = pd.read_csv('1Class3.txt', sep=" ", header = None)

x1 = x_one.iloc[:, :2].values
x2 = x_two.iloc[:, :2].values
x3 = x_three.iloc[:, :2].values

y1 = np.full(500, 1) # class-1
y2 = np.full(500, 2) # class-2
y3 = np.full(500, 3) # class-3

total_size = 1125

x_one = pd.read_csv('2class1.txt', sep=" ", header = None)
x_two = pd.read_csv('2class2.txt', sep=" ", header = None)
x_three = pd.read_csv('2class3.txt', sep=" ", header = None)

x1 = x_one.iloc[:, :2].values
x2 = x_two.iloc[:, :2].values
x3 = x_three.iloc[:, :2].values

y1 = np.full(2454, 1) # class-1
y2 = np.full(2488, 2) # class-2
y3 = np.full(2291, 3) # class-3

total_size = 5424

"""

# plotting the dataset
plt.scatter(x1_train[:, 0], x1_train[:, 1], color = 'red', alpha = 0.4)
plt.scatter(x2_train[:, 0], x2_train[:, 1], color = 'green', alpha = 0.4)
plt.xlabel('x1 feature')
plt.ylabel('x2 feature')
plt.show()

plt.scatter(x2_train[:, 0], x2_train[:, 1], color = 'green', alpha = 0.4)
plt.scatter(x3_train[:, 0], x3_train[:, 1], color = 'blue', alpha = 0.4)
plt.xlabel('x1 feature')
plt.ylabel('x2 feature')
plt.show()

plt.scatter(x3_train[:, 0], x3_train[:, 1], color = 'blue', alpha = 0.4)
plt.scatter(x1_train[:, 0], x1_train[:, 1], color = 'red', alpha = 0.4)
plt.xlabel('x1 feature')
plt.ylabel('x2 feature')
plt.show()



# declaring classifier handling objects
obj_1 = BayesClassifier(1, 2, x1_train, y1_train, x2_train, y2_train, total_size)
obj_2 = BayesClassifier(2, 3, x2_train, y2_train, x3_train, y3_train, total_size)
obj_3 = BayesClassifier(3, 1, x3_train, y3_train, x1_train, y1_train, total_size)



# preparing the meshgrid to plot the decision boundary
# Note: that for dataset 1, 2 use step = 0.1 and for dataset 3 use step = 5
x_set = np.concatenate((x1_train, x2_train, x3_train), axis = 0)
X1, X2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1,
                               stop = x_set[:, 0].max() + 1, step = 0.1),
np.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1,
          step = 0.1))

# plotting the decision surface between two classes
# choose the classifier according to the four cases of covariance matrix
# by default classifier_one is used in the below code 

plt.contourf(X1, X2, obj_1.classifier_one(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
            alpha = 0.4, cmap = ListedColormap(('red', 'green')))

plt.scatter(x1_train[:, 0], x1_train[:, 1], color = 'red', alpha = 1)
plt.scatter(x2_train[:, 0], x2_train[:, 1], color = 'green', alpha = 1)
plt.xlabel('x1 feature')
plt.ylabel('x2 feature')
plt.show()

plt.contourf(X1, X2, obj_2.classifier_one(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
            alpha = 0.4, cmap = ListedColormap(('green', 'blue')))

plt.scatter(x2_train[:, 0], x2_train[:, 1], color = 'green', alpha = 1)
plt.scatter(x3_train[:, 0], x3_train[:, 1], color = 'blue', alpha = 1)
plt.xlabel('x1 feature')
plt.ylabel('x2 feature')
plt.show()

plt.contourf(X1, X2, obj_3.classifier_one(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
            alpha = 0.4, cmap = ListedColormap(('red', 'blue')))

plt.scatter(x3_train[:, 0], x3_train[:, 1], color = 'blue', alpha = 1)
plt.scatter(x1_train[:, 0], x1_train[:, 1], color = 'red', alpha = 1)
plt.xlabel('x1 feature')
plt.ylabel('x2 feature')
plt.show()

# plotting global decision boundary between all the three classes together with
# their contour plots superimposed on them



glob_obj = GlobalClassifier(x1_train, y1_train, x2_train, y2_train, x3_train, y3_train, total_size)
plt.contourf(X1, X2, glob_obj.global_classifier_one(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
           alpha = 0.4, cmap = ListedColormap(('red', 'green', 'blue')))
plt.scatter(x1_train[:, 0], x1_train[:, 1], color = 'red', alpha = 1)
plt.scatter(x2_train[:, 0], x2_train[:, 1], color = 'green', alpha = 1)
plt.scatter(x3_train[:, 0], x3_train[:, 1], color = 'blue', alpha = 1)
plt.contour(X1, X2, obj_1.contour_func_i(X1, X2, 1), [0.3, 0.4, 0.5, 0.6])
plt.contour(X1, X2, obj_1.contour_func_j(X1, X2, 1), [0.3, 0.4, 0.5, 0.6])
plt.contour(X1, X2, obj_2.contour_func_j(X1, X2, 1), [0.3, 0.4, 0.5, 0.6])
plt.xlabel('x1 feature')
plt.ylabel('x2 feature')
plt.show()

plt.contourf(X1, X2, glob_obj.global_classifier_two(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
           alpha = 0.4, cmap = ListedColormap(('red', 'green', 'blue')))
plt.scatter(x1_train[:, 0], x1_train[:, 1], color = 'red', alpha = 1)
plt.scatter(x2_train[:, 0], x2_train[:, 1], color = 'green', alpha = 1)
plt.scatter(x3_train[:, 0], x3_train[:, 1], color = 'blue', alpha = 1)
plt.contour(X1, X2, obj_1.contour_func_i(X1, X2, 3), [0.6, 0.7, 0.8, 0.9])
plt.contour(X1, X2, obj_1.contour_func_j(X1, X2, 3), [0.6, 0.7, 0.8, 0.9])
plt.contour(X1, X2, obj_2.contour_func_j(X1, X2, 3), [0.3, 0.4, 0.5, 0.6])
plt.xlabel('x1 feature')
plt.ylabel('x2 feature')
plt.show()

plt.contourf(X1, X2, glob_obj.global_classifier_three(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
           alpha = 0.4, cmap = ListedColormap(('red', 'green', 'blue')))
plt.scatter(x1_train[:, 0], x1_train[:, 1], color = 'red', alpha = 1)
plt.scatter(x2_train[:, 0], x2_train[:, 1], color = 'green', alpha = 1)
plt.scatter(x3_train[:, 0], x3_train[:, 1], color = 'blue', alpha = 1)
plt.contour(X1, X2, obj_1.contour_func_i(X1, X2, 3), [0.6, 0.7, 0.8, 0.9])
plt.contour(X1, X2, obj_1.contour_func_j(X1, X2, 3), [0.6, 0.7, 0.8, 0.9])
plt.contour(X1, X2, obj_2.contour_func_j(X1, X2, 3), [0.3, 0.4, 0.5, 0.6])
plt.xlabel('x1 feature')
plt.ylabel('x2 feature')
plt.show()

plt.contourf(X1, X2, glob_obj.global_classifier_four(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
           alpha = 0.4, cmap = ListedColormap(('red', 'green', 'blue')))
plt.scatter(x1_train[:, 0], x1_train[:, 1], color = 'red', alpha = 1)
plt.scatter(x2_train[:, 0], x2_train[:, 1], color = 'green', alpha = 1)
plt.scatter(x3_train[:, 0], x3_train[:, 1], color = 'blue', alpha = 1)
plt.contour(X1, X2, obj_1.contour_func_i(X1, X2, 4), [0.6, 0.7, 0.8, 0.9])
plt.contour(X1, X2, obj_1.contour_func_j(X1, X2, 4), [0.6, 0.7, 0.8, 0.9])
plt.contour(X1, X2, obj_2.contour_func_j(X1, X2, 4), [0.3, 0.4, 0.5, 0.6])
plt.xlabel('x1 feature')
plt.ylabel('x2 feature')
plt.show()
