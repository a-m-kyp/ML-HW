import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import train_test_split

class SVM:

    def __init__(self, kernel, data, label, C=1, tolerance=1e-4, max_iter=100, polynomial_degree=2, sigma=1, epsilon=1e-4):
        self.data = data	                # X (MxN)
        self.label = label                  # y (Mx1)
        self.kernel = kernel                # Kernel function

        self.w = np.zeros((1,self.data.shape[1])) # weight (1xN)
        self.b = 0                          # bias
        self.h = {'w': self.w, 'b': self.b}      # hypothesis

        self.C = C		                    # upper bound parameter
        self.lagrange_multipliers = np.zeros(len(self.data))  # Lagrange multipliers lambda (N,)
        self.tolerance = tolerance          # tolerance
        self.max_iter = max_iter            # max iteration

        self.epsilon = epsilon              # epsilon
        self.polynomial_degree = polynomial_degree  # polynomial degree for polynomial kernel  
        self.sigma = sigma                  # sigma value for gaussian kernel

        self.k = data.dot(data.T)           # Kernel (MxM)
        
        """
                        [ Kernel(x1, Xj) ]	 [ Kernel(x1, x1) Kernel(x1, x2) .... Kernel(x1, xm) ]
        Kernel(Xi,Xj) = [	.... ....	 ] = [   .... ....	    .... ....	 ....   .... ....	 ]
                        [ Kernel(xm, Xj) ]	 [ Kernel(xm, x1) Kernel(xm, x2) .... Kernel(xm, xm) ]

        Kernel(xk,Xj) = [ Kernel(xk, x1) .... Kernel(xk, xm) ]

                        [ Kernel(x1, xk) ]
        Kernel(Xi,xk) = [	.... ....	 ]
                        [ Kernel(xm, xk) ]

        Xi, Xj: data[1:m,:]   or   xk	: data[k,:]
        Return: (m,m) matrix | (m, ) matrix
        """

    def hypothesis_sign(self, data):
        guess = np.inner(self.w, data) + self.b
        pos = (1)  * (guess > 0)
        neg = (-1) * (guess < 0)
        self.predict_label = pos + neg
        return self.predict_label
    
    def hypothesis_error(self, data, y_true):
        return np.mean(self.hypothesis_sign(data) != y_true)
    
    def hypothesis_update(self, w, b):
        self.w = w
        self.b = b

    def linear_kernel(self, Xi, Xj):
        """
        Linear Kernel: K(x,y) = x^T * y
        """
        return Xi.dot(Xj.T)

    def polynomial_kernel(self, Xi, Xj, polynomial_degree=2):
        """
        Polynomial Kernel: K(x,y) = (x^T * y + 1)^d
        """
        return (np.inner(Xi, Xj) + 1) ** polynomial_degree

    def gaussian_kernel(self, Xi, Xj, sigma=1):
        """
        Gaussian Kernel: K(x,y) = exp(-||x-y||^2 / (2 * sigma^2))
        """
        return np.exp(-0.5 * np.inner((Xi - Xj),(Xi - Xj)) / (1.0 * self.sigma ** 2))

    def calculate_kernel(self, X, Y):
        if self.kernel == 'linear':
            return self.linear_kernel(X, Y)
        elif self.kernel == 'polynomial':
            return self.polynomial_kernel(X, Y, self.polynomial_degree)
        elif self.kernel == 'gaussian':
            return self.gaussian_kernel(X, Y, self.sigma)
        else:
            raise ValueError('Kernel function not supported')

    def hypothesis(self, index):
        # self.k = self.calculate_kernel(self.data, self.data)
        # return (self.lagrange_multipliers * self.label.T).dot(self.k[index,:]) + self.b
        self.k = self.calculate_kernel(self.data, self.data)
        return (self.lagrange_multipliers * self.label.T).dot(self.k[index,:]) + self.b

    def fit(self):
        # Train the model
        num_sample, num_feature = self.data.shape
        # self.lagrange_multipliers = np.zeros_like(self.label, dtype=np.float64).ravel()
        passes = 0
        num_changed = 0 # number of changed lagrange multipliers
        examine_all = True # examine all data

        # self.k = self.calculate_kernel(self.data, self.data) # Kernel (MxM)

        threshold = len(self.label) * self.epsilon / self.tolerance # threshold for stopping criterion

        while (num_changed > threshold or examine_all) and passes < self.max_iter:
            passes += 1
            num_changed = 0

            if examine_all: 
                for j in range(num_sample):
                    num_changed += self.smo(j)
                    print("epoch: {}, num_changed: {}".format(passes, num_changed))
            else:
                # a vector of boolean values indicating whether the corresponding lambda is out of (0, C) because the corresponding support vector is not in the margin 
                out_of_bound = (self.lagrange_multipliers > 0) + (self.lagrange_multipliers < self.C)
                out_of_bound_index = np.arange(len(self.lagrange_multipliers))[out_of_bound]
                for j in out_of_bound_index:
                    num_changed += self.smo(j)
                    print("epoch: {}, num_changed: {}".format(passes, num_changed))
            print("[epoch %d] num_changed: %d" % (passes, num_changed))
            print("[epoch %d] Error: %f" % (passes, self.hypothesis_error(self.data, self.label)))
            if examine_all:
                examine_all = False
            elif num_changed <= threshold:
                examine_all = True

        else:
            print('Training finished')
            return self.w, self.b, self.lagrange_multipliers

    """" ### Sequential Minimal Optimization
    C: regularization parameter
    tolerance: tolerance for stopping criterion
    max_passes: maximum # of times to iterate over alpha without changing
    alpha: Lagrange multipliers
    b: threshold
    return lagrange_multipliers, bias
    """
    def smo(self, j):
        y_j= self.label[j]
        alpha_j = self.lagrange_multipliers[j]
        E_j = self.hypothesis(j) - y_j
        # print('E_j: ', E_j, 'alpha_j: ', alpha_j, 'C: ', self.C, 'tolerance: ', self.tolerance)
        if (y_j * E_j < -self.tolerance and alpha_j < self.C) or (y_j * E_j > self.tolerance and alpha_j > 0):
            out_of_bound = (self.lagrange_multipliers < 0) + (self.lagrange_multipliers > self.C)
            if (np.sum(out_of_bound) > 1):
                #  minus label.reshape(-1) is to make sure that the shape of E is (n,)
                self.k = self.calculate_kernel(self.data, self.data)
                error = (self.lagrange_multipliers * self.label.T).dot(self.k[:,:]) + self.b - self.label.reshape(-1)
                i_min = np.argmin(error)
                i_max = np.argmax(error)
                i = i_max if E_j <= 0 else i_min
                if self.take_each_step(i, j):
                    return 1
            out_of_bound_index = np.arange(len(self.lagrange_multipliers))[out_of_bound] # shape (n,)
            np.random.shuffle(out_of_bound_index)
            for i in out_of_bound_index:
                if self.take_each_step(i, j):
                    return 1
            all_index = np.arange(len(self.lagrange_multipliers))
            np.random.shuffle(all_index)
            for i in all_index:
                if self.take_each_step(i, j):
                    return 1
        return 0

    def take_each_step(self, i, j):
        """
        Sequential Minimal Optimization
        """
        if i == j:
            return False
        # calculate Ej
        
        y_i = self.label[i, 0]
        y_j = self.label[j, 0]

        Ei = self.hypothesis(i) - y_i
        Ej = self.hypothesis(j) - y_j
        
        alpha_i_old = self.lagrange_multipliers[i]
        alpha_j_old = self.lagrange_multipliers[j]

        sign = y_i * y_j
        # calculate L and H for alpha_j
        '''
        x_i     X_j
        -1      -1   -> if label[i] == label[j] = 1  where  -1 * -1 =  1
         1       1   -> if label[i] == label[j] = 1  where   1 *  1 =  1
        -1       1   -> if label[i] != label[j] = -1 where  -1 *  1 = -1
         1      -1   -> if label[i] != label[j] = -1 where   1 * -1 = -1
        
        So; sign = y_i * y_j
        if sign == 1 means (y_i == y_j) is True
        if sign == -1 means (y_i != y_j) is True

        So, we interpret sign as:
                                "sign > 0" or " 1"  as  True
                                "sign < 0" or "-1"  as  False
        '''
        if sign < 0:
            L = max(0, alpha_j_old - alpha_i_old)
            H = min(self.C, self.C + alpha_j_old - alpha_i_old)
        else:
            L = max(0, alpha_j_old + alpha_i_old - self.C)
            H = min(self.C, alpha_j_old + alpha_i_old)

        if L == H:
            return False

        # calculate eta
        # k_ij = self.calculate_kernel(self.data[i,:], self.data[j,:])
        # k_ii = self.calculate_kernel(self.data[i,:], self.data[i,:])
        # k_jj = self.calculate_kernel(self.data[j,:], self.data[j,:])
        self.k = self.calculate_kernel(self.data, self.data)
        k_ij = self.k[i, j]
        k_ii = self.k[i, i]
        k_jj = self.k[j, j]
        eta = 2 * k_ij - k_ii - k_jj
        print('eta: ', eta)

        # if eta > 0, clip alpha_j to L and H
        # XXX for eta == 0 we have overflow problem
        # "RuntimeWarning: invalid value encountered in true_divide" will be raised
        # need to handle it
        alpha_j = self.lagrange_multipliers[j]
        if eta > 0:
            self.lagrange_multipliers[j] = np.clip(alpha_j + y_j * (Ei - Ej) / eta, L, H)
        else:
            # XXX
            lower_bound = y_j * (Ei - Ej) * L
            upper_bound = y_j * (Ei - Ej) * H

            if upper_bound - lower_bound < 0.001:
                self.lagrange_multipliers[j] = L
            elif lower_bound - upper_bound > 0.001:
                self.lagrange_multipliers[j] = H
            else:
                return False

        if np.abs(self.lagrange_multipliers[j] - alpha_j_old) < self.epsilon:
            return False

        # update alpha_i
        self.lagrange_multipliers[i] += (y_i * y_j) * (alpha_j_old - self.lagrange_multipliers[j])

        # update b
        b1 = self.b - Ei - y_i * (self.lagrange_multipliers[i] - alpha_i_old) * k_ii - y_j * (self.lagrange_multipliers[j] - alpha_j_old) * k_ij 
        b2 = self.b - Ej - y_i * (self.lagrange_multipliers[i] - alpha_i_old) * k_ij - y_j * (self.lagrange_multipliers[j] - alpha_j_old) * k_jj
        
        if 0 < self.lagrange_multipliers[i] < self.C:
            self.b = b1
        elif 0 < self.lagrange_multipliers[j] < self.C:
            self.b = b2
        else:
            self.b = (b1 + b2) / 2

        # we update w by calculating the sum of all lagrange multipliers => w = sum(alpha_i * y_i * x_i)
        self.w = self.w + y_i * (self.lagrange_multipliers[i] - alpha_i_old) * self.data[i] + y_j * (self.lagrange_multipliers[j] - alpha_j_old) * self.data[j]
        return True

    def predict(self, data):
        """
        Predict the label of the data
        """
        return self.hypothesis_sign(data)

if __name__ == '__main__':
    # load data
    dataset = pd.read_csv("d1.csv", header=None)
    X = dataset.iloc[:, :-1]
    y = dataset.iloc[:, -1]

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # remap labels to {-1, 1}
    y_train = np.where(y_train == 0, -1, 1)
    y_test = np.where(y_test == 0, -1, 1)

    print("test data size: ", x_test.shape[0], "y_test size: ", y_test.shape[0])
    print("train data size: ", x_train, "y_train size: ", y_train)

    # Initialization of  C, tolerance, max_iter
    C, tolerance, max_iter, sigma, polynomial_degree = 1, 1e-4, 1000, 1, 2

    svm = SVM('linear', x_train, y_train, C=1, tolerance=1e-4, max_iter=1000, sigma=1)
    # svm = SVM('linea', x_train, y_train, C, tolerance, max_iter, sigma)
    # fit the model
    svm.fit()
    print("w: ", svm.w, "b: ", svm.b, "lagrange_multipliers: ", svm.lagrange_multipliers)
    # predict
    y_pred = svm.predict(x_test)
    print(y_pred)

    # print("[epoch %s] numChanged: %s" % (epoch, numChanged))
    # print("[epoch %s] Error: %s" % (epoch, h.error(ycv, xcv)))



    # def sequential_minimal_optimization(self):
    #     """" ### Sequential Minimal Optimization
    #     C: regularization parameter
    #     tolerance: tolerance for stopping criterion
    #     max_passes: maximum # of times to iterate over alpha without changing
    #     alpha: Lagrange multipliers
    #     b: threshold
    #     return lagrange_multipliers, bias
    #     """
    #     # initialize alpha, b and passes
    #     num_sample, num_feature = self.data.shape # num_sample is the number of samples
    #     self.alpha = np.zeros_like(self.label, dtype=np.float64)
    #     self.b = 0
    #     passes = 0 # passes is the number of passes through the data

    #     # loop until the number of passes is greater than max_passes or the difference between the current error and the previous error is less than the tolerance
    #     while passes < self.max_passes:
    #         num_changed_alphas = 0 # count the number of alphas that changed

    #         for i in range(num_sample): # for each training sample

    #             ############################################################
    #             #### Calculate Error[i] = f(x^(i)) - y^(i) where f(x^(i)) is the hypothesis -> f(x) = sum(alpha_i * y^(i) * K(x^(i), x)) + b
    #             ############################################################
    #             E_i = self.calculate_error(self.data[i], self.label[i], self.alpha[i])
    #             E_i = self.label[i] * E_i

    #             ############################################################
    #             #### If label[i]*E_i < -tolerance and alpha_i < C or if label[i]*E_i > tolerance and alpha_i > 0
    #             ############################################################
    #             if ((self.label[i] * E_i < -self.tolerance and self.alpha[i] < self.C) or (self.label[i] * E_i > self.tolerance and self.alpha[i] > 0)):

    #                 # Select j != i randomly.
    #                 j = i # j is the index of the sample that will be changed
    #                 while j == i:
    #                     j = np.random.choice(num_sample)

    #                 ############################################################
    #                 ## Calculate Error[j] = f(x^(j)) - y^(j) where f(x^(j)) is the hypothesis -> f(x) = sum(alpha_i * y^(i) * K(x^(i), x)) + b
    #                 ############################################################
    #                 E_j = self.calculate_error(self.data[j], self.label[j], self.alpha[j])
    #                 E_j = self.label[j] * E_j

    #                 # Store old alpha_i, alpha_j
    #                 alpha_i_old = self.alpha[i]
    #                 alpha_j_old = self.alpha[j]

    #                 ############################################################
    #                 ## H, L: upper and lower bounds for alpha_sample_j (L <= alpha_j <= H)
    #                 ############################################################
    #                 if self.label[i] != self.label[j]:
    #                     L = max(0, self.alpha[j] - self.alpha[i])
    #                     H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
    #                 if self.label[i] == self.label[j]:
    #                     L = max(0, self.alpha[i] + self.alpha[j] - self.C)
    #                     H = min(self.C, self.alpha[i] + self.alpha[j])
                    
    #                 # If L = H then skip this pair
    #                 if L == H:
    #                     continue

    #                 ############################################################
    #                 ## Calculate eta = 2 * K(x^(i), x^(j)) - K(x^(i), x^(i)) - K(x^(j), x^(j))
    #                 ## We use eta to calculate the new value of alpha_j
    #                 ## eta = 0 if eta is not a number or if eta is a small negative number
    #                 ############################################################
    #                 # eta = 2 * self.k[i, j] - self.k[i, i] - self.k[j, j]
    #                 i_j_inner = np.inner(self.data[i], self.data[j])
    #                 i_i_inner = np.inner(self.data[i], self.data[i])
    #                 j_j_inner = np.inner(self.data[j], self.data[j])
    #                 eta = 2 * i_j_inner - i_i_inner - j_j_inner
                    
    #                 # if eta >= 0 then skip this pair
    #                 if eta >= 0:
    #                     continue

    #                 ############################################################
    #                 ## Calculate alpha_j^(new) = alpha_j^(old) - y^(j) * (E_i - E_j) / eta
    #                 ############################################################
    #                 self.alpha[j] -= self.label[j] * (E_i - E_j) / eta

    #                 # if alpha_new^(j) > H or alpha_new^(j) < L then set alpha_j^(new) = L or H else keep alpha_j^(new)
    #                 if self.alpha[j] > H:
    #                     self.alpha[j] = H
    #                 elif self.alpha[j] < L:
    #                     self.alpha[j] = L

    #                 # if alpha_new^(j) is within the bounds then skip to the next pair
    #                 if abs(self.alpha[j] - alpha_j_old) < self.epsilon:
    #                     continue

    #                 ############################################################
    #                 ## Calculate alpha_new^(i) = alpha_i^(old) + y^(i) * y^(j) * (alpha_j^(old) - alpha_j^(new))
    #                 ############################################################
    #                 self.alpha[i] += self.label[i] * self.label[j] * (alpha_j_old - self.alpha[j])

    #                 ############################################################
    #                 ## Update b to reflect change in alpha_i, alpha_j and to take into account the new margins
    #                 ## b is the threshold of the SVM 
    #                 ############################################################
    #                 b1 = self.b - E_i - self.label[i] * (self.alpha[i] - alpha_i_old) * np.inner(self.data[i], self.data[i]) - self.label[j] * (self.alpha[j] - alpha_j_old) * np.inner(self.data[i], self.data[j])
    #                 b2 = self.b - E_j - self.label[i] * (self.alpha[i] - alpha_i_old) * np.inner(self.data[i], self.data[j]) - self.label[j] * (self.alpha[j] - alpha_j_old) * np.inner(self.data[j], self.data[j])

    #                 ############################################################
    #                 ## Update b based on which one is closer to 0
    #                 ############################################################
    #                 if 0 < self.alpha[i] < self.C:
    #                     self.b = b1
    #                 elif 0 < self.alpha[j] < self.C:
    #                     self.b = b2
    #                 else:
    #                     self.b = (b1 + b2) / 2

    #                 ## Update num_changed_alphas
    #                 num_changed_alphas += 1 # num_changed_alphas is the number of alphas that changed

    #         ############################################################
    #         ## If no alpha_i, alpha_j have been updated then terminate
    #         ############################################################
    #         if num_changed_alphas == 0:
    #             passes += 1 # increment passes
    #         else:
    #             passes = 0 # reset passes to 0

    #     ## alphas are the coefficients of the support vectors
    #     ## b is the threshold of the SVM
    #     return self.alpha, self.b
