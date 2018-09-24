import numpy as np
import math
from numpy import linalg as LA
import matplotlib.pyplot as plt
import matplotlib.colors as ListedColormap
from numpy.linalg import inv

"""
The class has all the attribute which does not depend on the training examples
and are common in most of the calculations

The class has mainly two methods:
    fit - it finds the class of the data point passed as argument
    classifier - it returns an array of predicted values for the array of data
    points passed as arguments
    
    the subscript represents the case on covariance matrix according to the 
    assignment.
"""
class BayesClassifier:
    class_i = 0
    class_j = 0
    xi_train = np.array([])
    yi_train = np.array([])
    xj_train = np.array([])
    yj_train = np.array([])
    P_ci = 0
    P_cj = 1
    mu_i = np.array([])
    mu_j = np.array([])
    cov_matrix_i = np.array([])
    cov_matrix_j = np.array([])
    total_size = 0
    
    def __init__(self, class_i, class_j, xi_train, yi_train, xj_train, yj_train, total_size):
        self.class_i = class_i
        self.class_j = class_j
        self.xi_train = xi_train
        self.yi_train = yi_train
        self.xj_train = xj_train
        self.yj_train = yj_train
        self.P_ci = np.size(yi_train) / total_size
        self.P_cj = np.size(yj_train) / total_size
        self.mu_i = np.array([np.mean(xi_train[:, 0]), np.mean(xi_train[:, 1])])
        self.mu_j = np.array([np.mean(xj_train[:, 0]), np.mean(xj_train[:, 1])])
        self.cov_matrix_i = np.cov(np.transpose(xi_train))
        self.cov_matrix_j = np.cov(np.transpose(xj_train))
        self.total_size = total_size
        
    def fit_one(self, x_vec):
        cov = (self.cov_matrix_i[0][0]+self.cov_matrix_j[0][0]+self.cov_matrix_i[1][1]+self.cov_matrix_j[1][1])/4
        dev_term = cov*cov*math.log(self.P_ci/self.P_cj)*(self.mu_i - self.mu_j)/(LA.norm(self.mu_i-self.mu_j)*LA.norm(self.mu_i-self.mu_j))
        x0 = (self.mu_i + self.mu_j)/2 - dev_term
        w = (self.mu_i - self.mu_j)
        return np.dot(w, x_vec) - np.dot(w, x0)
    
    def classifier_one(self, X_test):
        value = []
        for x_vec in X_test:
            if (self.fit_one(x_vec) > 0):
                value.append(self.class_i)
            else:
                value.append(self.class_j)
        
        return np.array(value)
    
    
    def fit_two(self, x_vec):
        avg_cov = np.array([[(self.cov_matrix_i[0][0]+self.cov_matrix_j[0][0])/2, (self.cov_matrix_i[0][1]+self.cov_matrix_j[0][1])/2], 
                             [(self.cov_matrix_i[1][0]+self.cov_matrix_j[1][0])/2, (self.cov_matrix_i[1][1]+self.cov_matrix_j[1][1])/2]])
        dev_term = math.log(self.P_ci/self.P_cj)*(self.mu_i - self.mu_j)/(np.dot((self.mu_i - self.mu_j), inv(avg_cov).dot(self.mu_i - self.mu_j)))
        x0 = (self.mu_i + self.mu_j)/2 - dev_term
        w = inv(avg_cov).dot(self.mu_i - self.mu_j)
        return np.dot(w, x_vec) - np.dot(w, x0)
    
    def classifier_two(self, X_test):
        value = []
        for x_vec in X_test:
            if (self.fit_two(x_vec) > 0):
                value.append(self.class_i)
            else:
                value.append(self.class_j)
        
        return np.array(value)
    
    
    def fit_three(self, x_vec):
        dia_i = np.array([[self.cov_matrix_i[0][0], 0], [0, self.cov_matrix_i[1][1]]])
        W_i = -1/2 * inv(dia_i)
        w_i = inv(dia_i).dot(self.mu_i)
        w_i0 = -1/2 * np.dot(self.mu_i, w_i) - 1/2 * math.log(abs(np.linalg.det(dia_i))) + math.log(self.P_ci)
        g_i = np.dot(x_vec, W_i.dot(x_vec)) + np.dot(w_i, x_vec) + w_i0
        
        dia_j = np.array([[self.cov_matrix_j[0][0], 0], [0, self.cov_matrix_j[1][1]]])
        W_j = -1/2 * inv(dia_j)
        w_j = inv(dia_j).dot(self.mu_j)
        w_j0 = -1/2 * np.dot(self.mu_j, w_j) - 1/2 * math.log(abs(np.linalg.det(dia_j))) + math.log(self.P_cj)
        g_j = np.dot(x_vec, W_j.dot(x_vec)) + np.dot(w_j, x_vec) + w_j0
        
        return g_i - g_j
    
    
    def classifier_three(self, X_test):
        value = []
        for x_vec in X_test:
            if (self.fit_three(x_vec) > 0):
                value.append(self.class_i)
            else:
                value.append(self.class_j)
            
        return np.array(value)
    
    def fit_four(self, x_vec):
        W_i = -1/2 * inv(self.cov_matrix_i)
        w_i = inv(self.cov_matrix_i).dot(self.mu_i)
        w_i0 = -1/2 * np.dot(self.mu_i, w_i) - 1/2 * math.log(abs(np.linalg.det(self.cov_matrix_i))) + math.log(self.P_ci)
        g_i = np.dot(x_vec, W_i.dot(x_vec)) + np.dot(w_i, x_vec) + w_i0
        
        W_j = -1/2 * inv(self.cov_matrix_j)
        w_j = inv(self.cov_matrix_j).dot(self.mu_j)
        w_j0 = -1/2 * np.dot(self.mu_j, w_j) - 1/2 * math.log(abs(np.linalg.det(self.cov_matrix_j))) + math.log(self.P_cj)
        g_j = np.dot(x_vec, W_j.dot(x_vec)) + np.dot(w_j, x_vec) + w_j0
        
        return g_i - g_j
    
    
    def classifier_four(self, X_test):
        value = []
        for x_vec in X_test:
            if (self.fit_four(x_vec) > 0):
                value.append(self.class_i)
            else:
                value.append(self.class_j)
            
        return np.array(value)
    
    def contour_curve(self, X1, X2, cov_matrix):
        return (1/math.sqrt(2*3.1417*abs(np.linalg.det(cov_matrix))))*pow(2.71828, -(cov_matrix[0][0]*X2*X2
               + cov_matrix[1][1]*X1*X1 - 2*cov_matrix[0][1]*X1*X2)/2)
        
    def contour_func_i(self, x1, x2, case):
        if (case == 1):
            cov = (self.cov_matrix_i[0][0]+self.cov_matrix_j[0][0]+self.cov_matrix_i[1][1]+self.cov_matrix_j[1][1])/4
            cov_matrix = np.array([[cov, 0], [0, cov]])
        elif (case == 2):
            cov_matrix = np.array([[(self.cov_matrix_i[0][0]+self.cov_matrix_j[0][0])/2, (self.cov_matrix_i[0][1]+self.cov_matrix_j[0][1])/2], 
                             [(self.cov_matrix_i[1][0]+self.cov_matrix_j[1][0])/2, (self.cov_matrix_i[1][1]+self.cov_matrix_j[1][1])/2]])
        elif (case == 3):
            cov_matrix = np.array([[self.cov_matrix_i[0][0], 0], [0, self.cov_matrix_i[1][1]]])
        elif (case == 4):
            cov_matrix = self.cov_matrix_i
        
        X1 = x1 - self.mu_i[0]
        X2 = x2 - self.mu_i[1]
        
        return self.contour_curve(X1, X2, cov_matrix)
    
    def contour_func_j(self, x1, x2, case):
        if (case == 1):
            cov = (self.cov_matrix_i[0][0]+self.cov_matrix_j[0][0]+self.cov_matrix_i[1][1]+self.cov_matrix_j[1][1])/4
            cov_matrix = np.array([[cov, 0], [0, cov]])
        elif (case == 2):
            cov_matrix = np.array([[(self.cov_matrix_i[0][0]+self.cov_matrix_j[0][0])/2, (self.cov_matrix_i[0][1]+self.cov_matrix_j[0][1])/2], 
                             [(self.cov_matrix_i[1][0]+self.cov_matrix_j[1][0])/2, (self.cov_matrix_i[1][1]+self.cov_matrix_j[1][1])/2]])
        elif (case == 3):
            cov_matrix = np.array([[self.cov_matrix_j[0][0], 0], [0, self.cov_matrix_j[1][1]]])
        elif (case == 4):
            cov_matrix = self.cov_matrix_j
        
        X1 = x1 - self.mu_j[0]
        X2 = x2 - self.mu_j[1]
        
        return self.contour_curve(X1, X2, cov_matrix)
        


       