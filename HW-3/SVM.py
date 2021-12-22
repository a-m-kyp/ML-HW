import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import plotly.express as px
from sklearn.svm import SVC
from utils import train_test_split, confusion_matrix, classification_report

class SVM_custom:
    def __init__(self, data, label, b, alpha, weight,  C=1, tolerance=1e-4, max_passes=100, polynomial_degree=2, sigma=1, epsilon=1e-5, kernel="linear"):
        self.kernel = kernel                                # 'linear', 'guassian' # ToDo: 'polynomial', 'sigmoid', 'laplacian', 'Bessel', 'Anova'
        self.data = data                                    # x (MxN)
        self.label = label                                  # label (Mx1)
        self.weight = weight                                # weights (1xN)
        self.b = b		                                    # 
        self.C = C		                                    # C is the penalty parameter of the error term
        self.sigma = sigma                                  # gaussian kernel parameter
        self.epsilon = epsilon                              # epsilon used in the stopping criteria
        self.tolerance = tolerance                          # tolerance 
        self.p_degree = polynomial_degree                   # polynomial kernel parameter
        self.max_passes = max_passes           
        self.alpha = alpha
        self.k = self.calculate_kernel(kernel)              # Kernel (MxM)

    def change_dataset(self, data, label):
        self.data = data
        self.label = label

    def linear_kernel(self, Xi, Xj):
        return Xi.dot(Xj.T)

    def poly_kernel(self, Xi, Xj, p_degree):
        return (np.dot(Xi, Xj.T) + 1) ** p_degree

    def gaussian_kernel(self, x, z, sigma, axis=None):
        return np.exp((-(np.linalg.norm(x-z, axis=axis)**2))/(2*sigma**2))
    
    def gaussian_matrix(self, X, sigma):
        row, col = X.shape
        gauss_matrix = np.zeros(shape=(row, row))
        X = np.asarray(X)
        i = 0
        for v_i in X:
            j = 0
            for v_j in X:
                gauss_matrix[i, j] = self.gaussian_kernel(v_i.T, v_j.T, sigma)
                j += 1
            i += 1
        return gauss_matrix

    def calculate_kernel(self, kernel):
        if kernel == "linear":
            return self.linear_kernel(self.data, self.data)
        elif kernel == "polynomial":
            # return self.poly_kernel(self.data[None,:, :], self.data[:,None, :], self.p_degree)
            return self.poly_kernel(self.data, self.data, self.p_degree)
        elif kernel == "gaussian":
            return self.gaussian_matrix(self.data, self.sigma)
        else:
            raise ValueError("Kernel type not supported")

    def hypothesis(self, i):
        return (self.alpha * self.label.T).dot(self.k[i, :]) + self.b


    def sequential_minimal_optimization(self):
        """" ### Sequential Minimal Optimization
        C: regularization parameter
        tolerance: tolerance for stopping criterion
        max_passes: maximum # of times to iterate over alpha without changing
        alpha: Lagrange multipliers
        b: threshold
        return self.lagrange_multipliers, bias
        """
        # initialize alpha, b, passes, weight
        num_sample, num_feature = self.data.shape 
        passes = 0
        while passes < self.max_passes:
            num_changed_alphas = 0 

            for i in range(num_sample): # for each training sample

                ############################################################
                #### Calculate E_i = f(x^(i)) - y^(i)
                #### Where f(x^(i)) is the hypothesis
                #### f(x) = alpha * y * k(x^(i), x) + b
                ############################################################
                y_i = self.label[i]
                alpha_i = self.alpha[i]
                E_i = self.hypothesis(i) - y_i

                # print('alpha_i = %s' % alpha_i, 'y_i = %s' % y_i, 'H(x^(i)) = %s' % self.hypothesis(i))
                # print("E_i: ", E_i, "\nalpha_i: ", alpha_i, "\ny_i: ", y_i)


                ############################################################
                #### if y^(i) * E_i < -tolerance and alpha_i < C \
                #### or
                #### y^(i) * E_i > tolerance and alpha_i > 0
                ############################################################

                if ((y_i * E_i < -self.tolerance and alpha_i < self.C) or (y_i * E_i > self.tolerance and alpha_i > 0)):

                    # Select j != i randomly.
                    j = i # j is the index of the sample that will be changed
                    while j == i:
                        j = np.random.choice(num_sample)

                    # print("i, j: ", i, j)

                    ############################################################
                    #### Calculate Error[j] = f(x^(j)) - y^(j)
                    #### Where f(x^(j)) is the hypothesis
                    #### f(x) = alpha * y * k(x^(j), x) + b
                    ############################################################
                    y_j = self.label[j]
                    E_j = self.hypothesis(j) - y_j
                    # print("E_j: ", E_j, "E_i: ", E_i)

                    alpha_i_old = self.alpha[i].copy()
                    alpha_j_old = self.alpha[j].copy()

                    ############################################################
                    ## H, L: upper and lower bounds for alpha_sample_j (L <= alpha_j <= H)
                    ############################################################
                    
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

                    sign =  y_i * y_j
                    if sign < 0 : # y_i != y_j
                        L = max(0, self.alpha[j] - self.alpha[i])
                        H = min(self.C, self.C + self.alpha[j] - self.alpha[i])                   
                    else:         # y_i == y_j
                        L = max(0, self.alpha[i] + self.alpha[j] - self.C)
                        H = min(self.C, self.alpha[i] + self.alpha[j])

                    # If L = H then skip this pair
                    if L == H:
                        continue

                    ############################################################
                    #### We use eta to calculate the new value of alpha_j 
                    #### Calculate eta = 2 * kij - kii - kjj
                    ############################################################
                    kij, kii, kjj = self.k[i, j], self.k[i, i], self.k[j, j]
                    eta = 2 * kij - kii - kjj
                    
                    ############################################################
                    # if eta > 0, clip alpha_j to L and H
                    # XXX for eta == 0 we have overflow problem
                    # "RuntimeWarning: invalid value encountered in true_divide" will be raised
                    # need to handle it
                    ############################################################
                    # if (eta > 0):
                    #     # we clip the lagrange_multiplier_j_new to make sure it is in the range [L,H]
                    #     self.alpha[j] = np.clip(alpha_j + y_j * (E_i - E_j) / eta, L, H)

                    # else:
                    #     # XXX: eta == 0
                    #     # lower bound and upper bound are 
                    #     lower_bound = y_j * (E_i - E_j) * L
                    #     upper_bound = y_j * (E_i - E_j) * H

                    #     # print("low_bound = ", lower_bound, " upper_bound = ", upper_bound)

                    #     # We use epsilon to make sure that the lagrange_multiplier_j_new is in the range [L,H], if not, we assign it to L or H
                    #     if (upper_bound - lower_bound < self.epsilon):
                    #         self.alpha[j] = L
                    #     elif (lower_bound - upper_bound > self.epsilon):
                    #         self.alpha[j] = H
                    #     else:
                    #         self.alpha[j] = L + (H - L) / 2
                    #         continue
                    if eta >= 0:
                        print("eta>=0")
                        continue
                    
                    self.alpha[j] -= y_j * (E_i - E_j) / eta
                    if self.alpha[j] > H:
                        self.alpha[j] = H
                    elif self.alpha[j] < L:
                        self.alpha[j] = L

                    if (np.abs(self.alpha[j] - alpha_j_old) < self.epsilon):
                        print("alpha_j not moving enough")
                        continue

                    ############################################################
                    #### update alpha_i
                    #### alpha_i = alpha_i_old + y_i * y_j * (alpha_j_new - alpha_j_old)
                    ############################################################
                    self.alpha[i] += (y_i * y_j) * (alpha_j_old - self.alpha[j])

                    ############################################################
                    #### b is the threshold of the SVM
                    #### b1 =  -E_i - y_i * (alpha_i_new - alpha_i_old) * kii - y_j * (alpha_j_new - alpha_j_old) * kij - b
                    #### b2 =  -E_j - y_i * (alpha_i_new - alpha_i_old) * kij - y_j * (alpha_j_new - alpha_j_old) * kjj - b
                    #### Update b to reflect change in alpha_i, alpha_j and to take into account the new margins
                    ############################################################

                    b_one = -(E_i + y_i * (self.alpha[i] - alpha_i_old) * kii + y_j * (self.alpha[j] - alpha_j_old) * kij) + self.b
                    b_two = -(E_j + y_i * (self.alpha[i] - alpha_i_old) * kij + y_j * (self.alpha[j] - alpha_j_old) * kjj) + self.b

                    if (0 < self.alpha[i]) and (self.alpha[i] < self.C):
                        self.b = b_one
                        print("b = i ", self.b)
                    elif (0 < self.alpha[j]) and (self.alpha[j] < self.C):
                        self.b = b_two
                        print("b = j ", self.b)
                    else:
                        self.b = (b_one + b_two) / 2
                        print("b = else ", self.b)

                    y_i = self.label[i]
                    y_j = self.label[j]

                    alpha_i = self.alpha[i].copy()
                    alpha_j = self.alpha[j].copy()

                    # # self.weight += alpha_i * y_i * self.X[i] + alpha_j * y_j * self.X[j]
                    # self.weight += y_i * (alpha_i - alpha_i_old) * self.data[i] + y_j * (alpha_j - alpha_j_old) * self.data[j]
                    # self.weight = self.weight + self.label[i] * (self.lagrange_multiplier[i] - alpha_i_old) * self.data[i] + self.label[j] * (self.lagrange_multiplier[j] - alpha_j_old) * self.data[j]

                    self.weight = self.weight + y_i * (alpha_i - alpha_i_old) * self.data[i] + y_j * (alpha_j - alpha_j_old) * self.data[j]

                    # print("Update b = ", self.b)
                    # print("Update w = ", self.weight)

                    num_changed_alphas += 1 # num_changed_alphas is the number of alphas that changed
                    print("pass = %d i: %d, %d pairs changed" % (passes, i, num_changed_alphas)) 
            ############################################################
            ## If no alpha_i, alpha_j have been updated then terminate
            ############################################################
            if num_changed_alphas == 0:
                passes += 1 # increment passes
            else:
                passes = 0 # reset passes to 0
            print("SVM training iteration = %d finished" % passes)
        ## alphas are the coefficients of the support vectors
        ## b is the threshold of the SVM
        # return self.lagrange_multiplier, self.b, self.weight
        else:
            print("Training Finished")

    def fit(self):
        self.sequential_minimal_optimization()
        return self.alpha, self.b, self.weight

    def decision_function(self, data):
        return np.inner(data, self.weight) + self.b

    def predict(self, data):
        return np.sign(self.decision_function(data))

    def get_params(self):
        return {'kernel type': self.kernel, 'C': self.C, 'tolerance': self.tolerance, 'max_passes': self.max_passes, 'epsilon': self.epsilon}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def __repr__(self) -> str:
        represent = "SVM :: Kernel_type: {}\tC: {:.2f}\tTolerance: {:.2f}\tMax_passes: {}\tEpsilon: {:.2f}".format(self.kernel, self.C, self.tolerance, self.max_passes, self.epsilon)
        if self.kernel == 'linear':
            represent += "\tWeight: {}".format(self.weight)
        elif self.kernel == 'polynomial':
            represent += "\tDegree: {}\tWeight: {}".format(self.p_degree, self.weight)
        elif self.kernel == 'rbf' or self.kernel == 'gauassian':
            represent += "\tSigma: {:.2f}\tWeight: {}".format(self.sigma, self.weight)
        return represent

    def plot_decision_boundry_2d(self, model, axes, plt_title, data, label):
        plt.axes(axes)

        X = data.copy()
        y = label.copy()

        x_limit = [np.min(X[:, 0]), np.max(X[:, 0])]
        y_limit = [np.min(X[:, 1]), np.max(X[:, 1])]

        x_mesh, y_mesh = np.meshgrid(np.linspace(*x_limit, 300), np.linspace(*y_limit, 300))
        rgb = np.array([[210, 0, 0], [0, 0, 150]])/255.0 # rgb is the color of the points
        
        helper = model.decision_function(np.c_[x_mesh.ravel(), y_mesh.ravel()]).reshape(x_mesh.shape)
        z_model = helper.copy()
        print(model.get_params()) 

        plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
        plt.contour(x_mesh, y_mesh, z_model, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--']) # levels are the levels of the contour lines
        plt.contourf(x_mesh, y_mesh, np.sign(z_model.reshape(x_mesh.shape)), alpha=0.3, levels=2, cmap=ListedColormap(rgb), zorder=1)
        plt.title(plt_title)
        return helper

    def sign(self, data):
        w_T_x_plus_b = np.inner(data, self.weight) + self.b
        positive = (1) * (w_T_x_plus_b >= 0)
        negative = (-1) * (w_T_x_plus_b < 0)
        self.y_predict = positive + negative
        return self.y_predict

    def accuracy(self, y_true, x_test):
        return np.mean(y_true == self.sign(x_test))

if __name__ == '__main__':

    part_one, part_two, part_three = False, True, False

    if part_one:
        dataset = pd.read_csv('d1.csv', header=None)
        dataset.columns = ['x1', 'x2', 'y']
        X = dataset.iloc[:, :-1]
        y = dataset.iloc[:, -1]

        dataset['y'] = dataset['y'].replace(0,-1)
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)

        # reshape y_train and y_test to be 1D
        y_train = y_train.reshape(-1)
        y_test = y_test.reshape(-1)

        print("test data size: ", x_test.shape, "y_test size: ", y_test.shape)
        print("train data size: ", x_train, "y_train size: ", y_train)

        b = 0
        w = np.zeros((1, x_train.shape[1]))
        alpha = np.zeros((x_train.shape[0]))
        svm_linear = SVM_custom(data=x_train, label=y_train, kernel="linear", b=b, alpha=alpha, weight=w, C=1, tolerance=0.0001, max_passes=1000, epsilon=1e-5, sigma=1)
        alphas, bias, weight = svm_linear.fit()
        print("SVM _ Model: ", svm_linear)
        print("alphas: ", alphas)
        print("bias: ", bias)
        print("weight: ", weight[0])
        accuracy = svm_linear.accuracy(y_test, x_test)
        print("accuracy: ", accuracy)

        model = SVC(kernel='linear', verbose=True)
        model = model.fit(x_train, y_train)

        # plot decision boundary
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
        svm_linear.plot_decision_boundry_2d(svm_linear, axs[0], "SVM Custom", data=np.array(X), label=np.array(y))
        svm_linear.plot_decision_boundry_2d(model, axs[1], "SVM Scikit-learn", data=np.array(X), label=np.array(y))
        plt.show()
        
    if part_two:
        dataset = pd.read_csv('d2.csv', header=None, names=['x1', 'x2', 'y'])
        X = dataset.iloc[:, :-1]
        y = dataset.iloc[:, -1]
        dataset['y'] = dataset['y'].replace(0,-1)
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)

        y_train = y_train.reshape(-1)
        y_test = y_test.reshape(-1)

        b = 0
        w = np.zeros((1, x_train.shape[1]))
        alpha = np.zeros((x_train.shape[0]))
        svm_gausssian = SVM_custom(data=x_train, label=y_train, kernel="gaussian", b=b, alpha=alpha, weight=w, C=1, tolerance=0.001, max_passes=100, epsilon=1e-5, sigma=20)
        svm_gausssian.fit()

        model = SVC(kernel='rbf', gamma=10, verbose=True)
        model = model.fit(x_train, y_train)
        # plot decision boundary
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
        z=svm_gausssian.plot_decision_boundry_2d(svm_gausssian, axs[0], "SVM Custom", data=np.array(X), label=np.array(y))
        z1=svm_gausssian.plot_decision_boundry_2d(model, axs[1], "SVM Scikit-learn", data=np.array(X), label=np.array(y))

        plt.show()
        print("SVM _ Model: ", svm_gausssian)

        # save to txt file 
        with open('./z.txt', 'w') as f:
            for item in z:
                f.write("%s\n" % item)
        with open('./z1.txt', 'w') as f:
            for item in z1:
                f.write("%s\n" % item)



# y_predict = svm_linear.predict(x_test)
# print(svm_linear.score(y_test, y_predict))
# svm_linear.plot_decision_boundry(x_test, y_predict)

# sns.set()

# fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
# dec = svm_linear.plot_decision_boundry_2d(axes=axs[0])

# dec1 = svm_linear.plot_decision_boundry_2d(model=model, axes=axs[1])
# plt.show()
# print(dec[20], dec1[20])
