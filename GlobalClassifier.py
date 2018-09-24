import numpy as np
import math
from numpy import linalg as LA
import matplotlib.pyplot as plt
import matplotlib.colors as ListedColormap
from numpy.linalg import inv

"""
This class deals with all the calculations required for all the three classes 
combined (i.e calculating decision boundary, confusion matrix, and other performance
parameters)

"""

class GlobalClassifier:
    xi_train = np.array([])
    yi_train = np.array([])
    xj_train = np.array([])
    yj_train = np.array([])
    xk_train = np.array([])
    yk_train = np.array([])
    P_ci = 0
    P_cj = 1
    P_ck = 0
    mu_i = np.array([])
    mu_j = np.array([])
    mu_k = np.array([])
    cov_matrix_i = np.array([])
    cov_matrix_j = np.array([])
    cov_matrix_k = np.array([])
    total_size = 0
    
    def __init__(self, xi_train, yi_train, xj_train, yj_train, xk_train, yk_train, total_size):
        self.xi_train = xi_train
        self.yi_train = yi_train
        self.xj_train = xj_train
        self.yj_train = yj_train
        self.xk_train = xk_train
        self.yk_train = yk_train
        self.P_ci = np.size(yi_train) / total_size
        self.P_cj = np.size(yj_train) / total_size
        self.P_ck = np.size(yk_train) / total_size
        self.mu_i = np.array([np.mean(xi_train[:, 0]), np.mean(xi_train[:, 1])])
        self.mu_j = np.array([np.mean(xj_train[:, 0]), np.mean(xj_train[:, 1])])
        self.mu_k = np.array([np.mean(xk_train[:, 0]), np.mean(xk_train[:, 1])])
        self.cov_matrix_i = np.cov(np.transpose(xi_train))
        self.cov_matrix_j = np.cov(np.transpose(xj_train))
        self.cov_matrix_k = np.cov(np.transpose(xk_train))
        self.total_size = total_size
        
    def max(self, a, b, c):
        if (a > b):
           val = a
           Class = 1
        else:
           val = b
           Class = 2
        if (val < c):
            val = c
            Class = 3
            
        return Class
    
    def global_fit_one(self, x_vec):
        #
        cov = (self.cov_matrix_i[0][0]+self.cov_matrix_j[0][0]+self.cov_matrix_k[0][0]+self.cov_matrix_i[1][1]+self.cov_matrix_j[1][1]+self.cov_matrix_k[1][1])/6
        g_i = -(np.dot(x_vec - self.mu_i, x_vec - self.mu_i))/2*cov*cov + math.log(self.P_ci)
        g_j =  -(np.dot(x_vec - self.mu_j, x_vec - self.mu_j))/2*cov*cov + math.log(self.P_cj)
        g_k =  -(np.dot(x_vec - self.mu_k, x_vec - self.mu_k))/2*cov*cov + math.log(self.P_ck)
        
        return self.max(g_i, g_j, g_k)
    
    def global_classifier_one(self, X):
        value = []
        for x_vec in X:
            value.append(self.global_fit_one(x_vec))
            
        return np.array(value)
        
    def global_fit_two(self, x_vec):
        #avg_cov = np.array([[(self.cov_matrix_i[0][0]+self.cov_matrix_j[0][0]+self.cov_matrix_k[0][0]])/2, (self.cov_matrix_i[0][1]+self.cov_matrix_j[0][1]+self.cov_matrix_k[0][1])/2], [(self.cov_matrix_i[1][0]+self.cov_matrix_j[1][0]+self.cov_matrix_k[1][0])/2, (self.cov_matrix_i[1][1]+self.cov_matrix_j[1][1]+self.cov_matrix_k[1][1])/2]])
        
        cov1 = (self.cov_matrix_i[0][0] + self.cov_matrix_j[0][0] + self.cov_matrix_k[0][0])/2
        cov2 = (self.cov_matrix_i[0][1] + self.cov_matrix_j[0][1] + self.cov_matrix_k[0][1])/2
        cov3 = (self.cov_matrix_i[1][0] + self.cov_matrix_j[1][0] + self.cov_matrix_k[1][0])/2
        cov4 = (self.cov_matrix_i[1][1] + self.cov_matrix_j[1][1] + self.cov_matrix_k[1][1])/2
        
        avg_cov = np.array([[cov1, cov2], [cov3, cov4]])
        
        g_i = -(np.dot(x_vec - self.mu_i, inv(avg_cov).dot(x_vec - self.mu_i)))/2 + math.log(self.P_ci)
        g_j = -(np.dot(x_vec - self.mu_j, inv(avg_cov).dot(x_vec - self.mu_j)))/2 + math.log(self.P_cj)
        g_k = -(np.dot(x_vec - self.mu_k, inv(avg_cov).dot(x_vec - self.mu_k)))/2 + math.log(self.P_ck)
        
        return self.max(g_i, g_j, g_k)
        
    def global_classifier_two(self, X):
        value = []
        for x_vec in X:
            value.append(self.global_fit_two(x_vec))
            
        return np.array(value)
    
    def global_fit_three(self, x_vec):
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
        
        dia_k = np.array([[self.cov_matrix_k[0][0], 0], [0, self.cov_matrix_k[1][1]]])
        W_k = -1/2 * inv(dia_k)
        w_k = inv(dia_k).dot(self.mu_k)
        w_k0 = -1/2 * np.dot(self.mu_k, w_k) - 1/2 * math.log(abs(np.linalg.det(dia_k))) + math.log(self.P_ck)
        g_k = np.dot(x_vec, W_k.dot(x_vec)) + np.dot(w_k, x_vec) + w_k0
        
        return self.max(g_i, g_j, g_k)
    
    def global_classifier_three(self, X):
        value = []
        for x_vec in X:
            value.append(self.global_fit_three(x_vec))
            
        return np.array(value)
    
    def global_fit_four(self, x_vec):
        W_i = -1/2 * inv(self.cov_matrix_i)
        w_i = inv(self.cov_matrix_i).dot(self.mu_i)
        w_i0 = -1/2 * np.dot(self.mu_i, w_i) - 1/2 * math.log(abs(np.linalg.det(self.cov_matrix_i))) + math.log(self.P_ci)
        g_i = np.dot(x_vec, W_i.dot(x_vec)) + np.dot(w_i, x_vec) + w_i0
        
        W_j = -1/2 * inv(self.cov_matrix_j)
        w_j = inv(self.cov_matrix_j).dot(self.mu_j)
        w_j0 = -1/2 * np.dot(self.mu_j, w_j) - 1/2 * math.log(abs(np.linalg.det(self.cov_matrix_j))) + math.log(self.P_cj)
        g_j = np.dot(x_vec, W_j.dot(x_vec)) + np.dot(w_j, x_vec) + w_j0
        
        W_k = -1/2 * inv(self.cov_matrix_k)
        w_k = inv(self.cov_matrix_k).dot(self.mu_k)
        w_k0 = -1/2 * np.dot(self.mu_k, w_k) - 1/2 * math.log(abs(np.linalg.det(self.cov_matrix_k))) + math.log(self.P_ck)
        g_k = np.dot(x_vec, W_k.dot(x_vec)) + np.dot(w_k, x_vec) + w_k0
        
        return self.max(g_i, g_j, g_k)
    
    def global_classifier_four(self, X):
        value = []
        for x_vec in X:
            value.append(self.global_fit_four(x_vec))
            
        return np.array(value)     
        
        


       