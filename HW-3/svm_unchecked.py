import numpy as np
import pandas as pd
from copy import deepcopy
import matplotlib.pyplot as plt
from utils import train_test_split
from matplotlib.colors import ListedColormap

class SMO:
    def __init__(self, data, label, w, b=0, C=1, tolerance=1e-4, max_iter=100, polynomial_degree=2, sigma=1, epsilon=1e-4, kernel="linear"):
        self.kernel = kernel    # 'linear' or 'gaussian'
        self.data = data        # x (MxN)
        self.label = label      # y (Mx1)
        self.w = w		        # weight (1xN)
        self.b = b		        # bias
        self.C = C		        # upper bound parameter
        self.sigma = sigma      # gaussian kernel parameter
        self.epsilon = epsilon  # epsilon used in the stopping criteria
        self.tolerance = tolerance  # tolerance 
        self.p_degree = polynomial_degree  # polynomial kernel parameter
        self.max_iter = max_iter  # maximum iteration
        self.lagrange_multiplier = np.zeros(len(label))   # lagrange_multiplierbda (N,)
        self.k = self.calculate_kernel(kernel)  # Kernel (MxM)

    def calculate_kernel(self, kernel):
        if kernel == "linear":
            return self.linear_kernel(self.data, self.data)
        elif kernel == "gaussian":
            return self.gaussian_kernel(self.data[None, :, :], self.data[:, None, :], self.sigma, axis=2)
        else:
            raise ValueError("Kernel type not supported")

    def change_dataset(self, data, label):
        self.data = data
        self.label = label

    def linear_kernel(self, Xi, Xj):
        return Xi.dot(Xj.T)

    def gaussian_kernel(self, Xi, Xj, sigma, axis=None):
        return np.exp((-(np.linalg.norm(Xi-Xj, axis=axis)**2))/(2*sigma**2))

    def poly_kernel(self, Xi, Xj, degree):
        return (Xi.dot(Xj.T) + 1)**degree


    def H_x(self, i):
        return (self.lagrange_multiplier * self.label.T).dot(self.k[i, :]) + self.b

    def takeStep(self, i, j):
        if (i == j):
            return False  # continue
        eps = self.epsilon
        lagrange_multiplier_i = self.lagrange_multiplier[i]
        lagrange_multiplier_j = self.lagrange_multiplier[j]
        y1 = self.label[i, 0]
        y2 = self.label[j, 0]
        E1 = self.H_x(i) - y1
        E2 = self.H_x(j) - y2
        # print(E1, E2)
        # we need to check if s = -1 or 1 so that we can use sign(s)
        s = y1 * y2
        # s =  True if y1 != y2 else False

        # print(" --y1-- ", y1, " ---y2--- ", y2, " ---sign--- ", s)
        # print(" --y1-- ", y1, " ---y2--- ", y2, " ---sign--- ", s)

        L = max(0, lagrange_multiplier_j - lagrange_multiplier_i) if s < 0 else max(0,
                                                                                  lagrange_multiplier_j + lagrange_multiplier_i - self.C)
        # L	 = max(0, lagrange_multiplier_j - lagrange_multiplier_i) if s else max(0, lagrange_multiplier_j + lagrange_multiplier_i - self.C)

        H = min(self.C, self.C + lagrange_multiplier_j -
                lagrange_multiplier_i) if s < 0 else min(self.C, lagrange_multiplier_j + lagrange_multiplier_i)
        # H	 = min(self.C, self.C + lagrange_multiplier_j - lagrange_multiplier_i) if s else min(self.C, lagrange_multiplier_j + lagrange_multiplier_i)

        ''''
		Final Result:
		+ fold:  5
		+ Optimal C:  0.7
		+ optimal w:  [[-4379726.31225647 -5940667.69855967]]
		+ optimal b:  [2322539.30898146]
		+ Testing error:  0.5838150289017341
		'''
        '''
		Final Result:
		+ fold:  5
		+ Optimal C:  0.7
		+ optimal w:  [[-3.96136032 13.71842515]]
		+ optimal b:  [0.19742505]
		+ Testing error:  0.4161849710982659
		'''

        # print("L = ", L, " H = ", H)

        if L == H:
            return False
        K11 = self.k[i, i]
        K12 = self.k[i, j]
        K22 = self.k[j, j]
        eta = K11 + K22 - 2 * K12

        if (eta > 0):  # if eta = 0 then "RuntimeWarning: invalid value encountered in true_divide" will be raised
            # we clip the lagrange_multiplier_j_new to make sure it is in the range [L,H]
            lagrange_multiplier_j_new = np.clip(
                lagrange_multiplier_j + y2 * (E1 - E2) / eta, L, H)
        else:  # eta <= 0
            # y2 * (E1 - E2) * L
            lower_bound = y2 * (E1 - E2) * L
            # y2 * (E1 - E2) * H
            upper_bound = y2 * (E1 - E2) * H
            # print("low_bound = ", low_bound, " upper_bound = ", upper_bound)

            # we use eps to make sure that the lagrange_multiplier_j_new is in the range [L,H]
            # if upper_bound - lower_bound < eps then lagrange_multiplier_j_new = upper_bound then we can't find a value in the range [L,H]
            if (upper_bound - lower_bound < eps):
                lagrange_multiplier_j_new = L
            # if lower_bound - upper_bound > eps then lagrange_multiplier_j_new = lower_bound then we can't find a value in the range [L,H]
            elif (lower_bound - upper_bound > eps):
                lagrange_multiplier_j_new = H
            else:
                return False

        # * (lagrange_multiplier_j + lagrange_multiplier_j_new + eps)): # we multiply eps by (lagrange_multiplier_j + lagrange_multiplier_j_new + eps)
        if (abs(lagrange_multiplier_j - lagrange_multiplier_j_new) < eps):
            return False

        lagrange_multiplier_i_new = lagrange_multiplier_i + s * \
            (lagrange_multiplier_j - lagrange_multiplier_j_new)

        ########################################################################
        # Update b
        ########################################################################
        b1 = -(E1 + y1 * (lagrange_multiplier_i_new - lagrange_multiplier_i) * K11 +
               y2 * (lagrange_multiplier_j_new - lagrange_multiplier_j) * K12) + self.b
        b2 = -(E2 + y1 * (lagrange_multiplier_i_new - lagrange_multiplier_i) * K12 +
               y2 * (lagrange_multiplier_j_new - lagrange_multiplier_j) * K22) + self.b

        # if (0 < lagrange_multiplier_i_new and lagrange_multiplier_i_new < self.C):
        if (0 < lagrange_multiplier_i_new < self.C):
            self.b = b1
        # elif (0 < lagrange_multiplier_j_new and lagrange_multiplier_j_new < self.C):
        elif (0 < lagrange_multiplier_j_new < self.C):
            self.b = b2
        else:
            self.b = (b1 + b2) / 2

        # print("Update b = ", self.b)
        ########################################################################
        # Update w
        ########################################################################
        # we update w by adding the difference of the new lagrange_multiplier_i and lagrange_multiplier_j
        self.w = self.w + y1 * (lagrange_multiplier_i_new - lagrange_multiplier_i) * \
            self.data[i] + y2 * (lagrange_multiplier_j_new -
                                 lagrange_multiplier_j) * self.data[j]
        # print("Update w = ", self.w)

        ########################################################################
        # Update error cache without bias F
        ########################################################################
        # self.F = self.F + y1 * (lagrange_multiplier_i_new - lagrange_multiplier_i) * self.k[i,:] \
        # + y2 * (lagrange_multiplier_j_new - lagrange_multiplier_j) * self.k[j,:]

        ########################################################################
        # Update lagrange_multiplierbda_i and lagrange_multiplierbda_j
        ########################################################################
        self.lagrange_multiplier[i] = lagrange_multiplier_i_new
        self.lagrange_multiplier[j] = lagrange_multiplier_j_new
        # print("Update lagrange_multiplier[%s] = %s		(old: %s)" % (i, lagrange_multiplier_i_new, lagrange_multiplier_i))
        # print("Update lagrange_multiplier[%s] = %s		(old: %s)" % (j, lagrange_multiplier_j_new, lagrange_multiplier_j))
        return True

    def examine(self, j, tol=0.001):
        # print("choose j = ", j)
        # y2	 = self.label[j,0] # y2 is y_j
        y2 = self.label[j]  # y2 is y_j

        lagrange_multiplier_j = self.lagrange_multiplier[j]
        E2 = self.H_x(j) - y2  # E2 is E_j

        # print("E(%s) : %s" % (j, E2))
        r2 = E2 * y2

        if ((r2 < -tol and lagrange_multiplier_j < self.C) or (r2 > tol and lagrange_multiplier_j > 0)):
            ######
            # cached error is too big, must to update
            #####
            # we use this to find the indices of the out of bound elements, out_of_bound elements are the elements that are not in the range [0,C]
            out_of_bound = (self.lagrange_multiplier < 0) + \
                (self.lagrange_multiplier > self.C)
            if (np.sum(out_of_bound) > 1):
                # minus label.reshape(-1) is to make sure that the shape of E is (n,)
                E = (
                    self.lagrange_multiplier * self.label.T).dot(self.k[:, :]) + self.b - self.label.reshape(-1)
                # we are going to update the index of the element that is out of bound
                i = np.argmax(E) if E2 <= 0 else np.argmin(E)
                # if takeStep(i,j) returns True, then we update the index of the element that is out of bound
                if (self.takeStep(i, j)):
                    # print("///////  CASE1 : choose i = ", i)
                    # print("-----------UPDATE-----------")
                    return 1
            out_index = np.arange(len(self.lagrange_multiplier))[out_of_bound]
            np.random.shuffle(out_index)
            for i in out_index:
                if (self.takeStep(i, j)):
                    # print("///////  CASE2 : choose i = ", i)
                    # print("-----------UPDATE-----------")
                    return 1
            all_index = np.arange(len(self.lagrange_multiplier))
            np.random.shuffle(all_index)
            for i in all_index:
                if (self.takeStep(i, j)):
                    # print("///////  CASE3 : choose i = ", i)
                    # print("-----------UPDATE-----------")
                    return 1
        return 0


class Hypothesis:
    def __init__(self, w, b):
        self.w = w
        self.b = b

    def sgn(self, data):
        guess = data[:].dot(self.w.T) + self.b
        pos = (1) * (guess > 0)
        neg = (-1) * (guess < 0)
        self.label = pos + neg
        return self.label

    def error(self, c_label, data):
        h_label = self.sgn(data)
        return np.mean(c_label != h_label)

    def update(self, w, b):
        self.w = w
        self.b = b

if __name__ == '__main__':

    dataset = pd.read_csv("d1.csv", header=None)
    X = dataset.iloc[:, :-1]
    y = dataset.iloc[:, -1]

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


    # remap label to -1, +1 classes
    y_train = np.where(y_train == 0, -1, 1)
    y_test = np.where(y_test == 0, -1, 1)

    print("x_train : ", x_train.shape, "y_train : ", y_train.shape,
          "x_test : ", x_test.shape, "y_test : ", y_test.shape)

    # x_train, y_train = shuffle_union(x_train, y_train)
    # x_test, y_test = shuffle_union(x_test, y_test)
    # fold_size = len(x_train) // n
    # R_i = 0  # R_i is the number of correctly classified samples in the ith fold
    # R_opt = 1  # R_opt is the number of correctly classified samples in the optimal fold

    # print("x_train : ", x_train.shape, "y_train : ", y_train.shape,
    #       "x_test : ", x_test.shape, "y_test : ", y_test.shape)

    w = np.zeros((1, x_train.shape[1])) # weight: (1xN)
    b = 0                               # bias: (1x1)
    h = Hypothesis(w, b)                # Initializing h

    # for i in range():
        ###################################
        # Cross-validation training
        ###################################
        # xcv = np.concatenate(
        #     (x_train[:i * fold_size], x_train[(i+1) * fold_size:]), axis=0)  # xcv: (m, N)
        # ycv = np.concatenate(
        #     (y_train[:i * fold_size], y_train[(i+1) * fold_size:]), axis=0)  # ycv: (m, 1)
        # print("xcv.shape = ", xcv.shape, "ycv.shape = ", ycv.shape)

        # svm_model = SMO(xcv, ycv, w, b, C)
    svm_model = SMO(kernel="gaussian", data=x_train, label=y_train, w=w, b=0, C=1, sigma=5)

    numChanged = 0
    toCheckAll = True  # toCheckAll is a flag to check all training samples
    epoch = 0
        # threshold is different from tol because tol is the tolerance of support vector and threshold is the tolerance of the number of support vector
    threshold = len(x_train) * svm_model.epsilon / svm_model.tolerance
        # we stop when the number of support vector is less than the tolerance or we have checked all training samples
    while epoch < svm_model.max_iter:
    # while ((numChanged > threshold or toCheckAll) and epoch < svm_model.max_iter):
        epoch += 1
        numChanged = 0  # number of changed support vectors

        # if (toCheckAll):  # toCheckAll is a flag to check all training samples and not only the support vectors
        for j in range(len(x_train)):
            # examine the jth training sample
            numChanged += svm_model.examine(j, svm_model.tolerance)
            print("epoch = ", epoch, "numChanged = ", numChanged)
            # print("[epoch %s] numChanged: %s" % (epoch, numChanged))

        # else:  # then toCheckAll is False and we check only the support vectors

        #     # out_of_bound is a vector of boolean values indicating whether the corresponding lagrange_multiplierbda is out of [0, C] because the corresponding support vector is not in the margin
        #     out_of_bound = (svm_model.lagrange_multiplier > 0) + \
        #         (svm_model.lagrange_multiplier < svm_model.C)

        #     print(" > 0 ", svm_model.lagrange_multiplier[svm_model.lagrange_multiplier > 0],
        #             " < C ", svm_model.lagrange_multiplier[svm_model.lagrange_multiplier < svm_model.C].shape)
        #     print("\ntrainner.lagrange_multiplier = ", svm_model.lagrange_multiplier, "\nsvm_model.w = ",
        #             svm_model.w, "\nsvm_model.b = ", svm_model.b)
        #     print("Out of bound = ", out_of_bound)

        #     # we apply [out_of_bound] to get the indices of the out of bound support vectors
        #     out_index = np.arange(len(svm_model.lagrange_multiplier))[
        #         out_of_bound]

        #     print("out_index = ", out_index)

        #     for j in out_index:  # j is the index of out_of_bound weight
        #         # examine the jth training sample
        #         numChanged += svm_model.examine(j, svm_model.tolerance)
        #         # print("[epoch %s] numChanged: %s" % (epoch, numChanged))
        w = svm_model.w
        b = svm_model.b
        # lagrange_multiplier = svm_model.lagrange_multiplier
        # alpha = svm_model.label
        h.update(w, b)  # update the hypothesis

        # print the number of changed support vectors
        print("[epoch %s] numChanged: %s" % (epoch, numChanged))
        # print the error of the current hypothesis
        print("[epoch %s] Error: %s" % (epoch, h.error(y_train, x_train)))
        if (toCheckAll):  # if we checked all training samples, then we change the flag to check only the support vectors
            toCheckAll = False
        # if the number of changed support vectors is less than the threshold, then we change the flag to check all training samples
        elif (numChanged <= threshold):
            toCheckAll = True

    else:
        print("Training Finished!")
        h.update(svm_model.w, svm_model.b)
        print("w = ", svm_model.w, "b = ", svm_model.b)
        print("alphas = ", svm_model.lagrange_multiplier)

        ###################################
        # Cross-validation testing
        # ###################################
        # xcv = x_train[i * fold_size:(i+1) * fold_size]
        # ycv = y_train[i * fold_size:(i+1) * fold_size]
        # print("=============================================")
        # print("[%sth] CV-Test Result:" % i)
        # print("+ w: ", h.w)
        # print("+ b: ", h.b)
        # print("+ Testing error: ", h.error(ycv, xcv))
        # print("=============================================")

        # R_i += h.error(ycv, xcv)
    ###///// End of for(i in fold) /////###

    # R_cv = R_i / n  # R_cv is the average of the number of correctly classified samples in the n folds
    # if R_cv < R_opt:  # If the number of correctly classified samples in the n folds is less than the optimal number of correctly classified samples in the n folds, then update the optimal number of correctly classified samples in the n folds
    #     C_opt = C  # C_opt is the optimal value of C
    #     h_opt = deepcopy(h)  # h_opt is the optimal hypothesis

    # print("=============================================")
    # print("Final Result:")
    # print("+ fold: ", n)
    # print("+ Optimal C: ", C_opt)
    # print("+ optimal w: ", h_opt.w)
    # print("+ optimal b: ", h_opt.b)
    # print("+ Testing error: ", h.error(y_test, x_test))
    # print("=============================================")

    # print(h_opt.w, h_opt.b)

    # xlim = [np.min(x_train[:,0]), np.max(x_train[:,0])]
    # ylim = [np.min(x_train[:,1]), np.max(x_train[:,1])]
    # x_mesh, y_mesh = np.meshgrid(np.linspace(xlim[0], xlim[1], 100), np.linspace(ylim[0], ylim[1], 100))
    # rgb = np.array([[210,0,0], [0, 0, 150]]) / 255.0

    # calculate the decision boundary
    # z_model = np.zeros((100,100))
    # for i in range(x_mesh.shape[0]):
    # 	# z_model[i,:] = np.dot(h_opt.w, np.array([x_mesh[i,:], y_mesh[i,:]].ravel()))
    # 	z_model[i,:] = np.dot(h_opt.w, np.array([x_mesh[i,:], y_mesh[i,:]]))

    # z_model = z_model + h_opt.b

    # z_model = np.dot(h_opt.w, np.array([x_mesh, y_mesh]).T) + h_opt.b

    # plt.scatter(x_train[:,0], x_train[:,1], c=y_train, s=50, cmap='viridis')
    # plt.contour(x_mesh, y_mesh, z_model, colors='k', levels=[-1, 0, 1], linestyles=['solid', 'dashed', 'solid'])
    # plt.contourf(x_mesh, y_mesh, np.sign(z_model.reshape(x_mesh.shape)), alpha=0.3, levels=2, cmap=ListedColormap(rgb), zorder=1)
    # plt.show()

    # "\nh_opt.lagrange_multiplier = ", h_opt.lagrange_multiplier, "\nh_opt.alpha = ", h_opt.alpha)
    # print("h_opt.w = ", h_opt.w, "\nh_opt.b = ", h_opt.b)

    # fig, ax = plt.subplots()
    # X0, X1 = x_train[:,0], x_train[:,1]
    # xx, yy = make_meshgrid(X0, X1)
    # plot_contours(ax, h_opt.w, h_opt.b, xx, yy, cmap=plt.cm.coolwarm, alpha=0.3)
    # ax.scatter(X0, X1, c=y_train, cmap=plt.cm.coolwarm, s=50, edgecolors='k')
    # ax.set_ylabel('y')
    # ax.set_xlabel('x')
    # ax.set_title("Decision Boundary")
    # plt.show()
