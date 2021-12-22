from math import gamma
from os import name
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import svm
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.utils import validation
from utils import train_test_split, confusion_matrix, classification_report

sns.set()

class SVM_custom:
    def __init__(self, data, label, alpha, weight, b=0, C=1, tolerance=1e-4, max_passes=100, polynomial_degree=2, sigma=1, epsilon=1e-5, kernel="linear", verbose=False):
        self.kernel = kernel                                # 'linear', 'guassian' # ToDo: 'polynomial', 'sigmoid', 'laplacian', 'Bessel', 'Anova'
        self.data = data                                    # x (MxN)
        self.label = label                                  # label (Mx1)
        self.weight = weight                                # weights (1xN)
        self.b = b		                                    # bias (1x1)
        self.C = C                                          # C is the penalty parameter of the error term
        self.sigma = sigma                                  # gaussian kernel parameter
        self.epsilon = epsilon                              # epsilon used in the stopping criteria
        self.tolerance = tolerance                          # tolerance
        self.p_degree = polynomial_degree                   # polynomial kernel parameter  (degree)
        self.max_passes = max_passes                        # maximum number of passes over the training data
        self.alpha = alpha                                  # alpha (Mx1)
        self.k = self.calculate_kernel(kernel)              # Kernel (MxM)
        self.verbose = verbose                              # verbose

    def set_dataset(self, data, label):
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
            # return self.gaussian_kernel(self.data[None, :, :], self.data[:, None, :], self.sigma, axis=2)
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
        num_sample, num_feature = self.data.shape
        passes = 0
        while passes < self.max_passes:
            num_changed_alphas = 0

            for i in range(num_sample):  # for each training sample

                ############################################################
                # Calculate E_i = f(x^(i)) - y^(i)
                # Where f(x^(i)) is the hypothesis
                # f(x) = alpha * y * k(x^(i), x) + b
                ############################################################
                y_i = self.label[i]
                alpha_i = self.alpha[i]
                E_i = self.hypothesis(i) - y_i

                # print('alpha_i = %s' % alpha_i, 'y_i = %s' % y_i, 'H(x^(i)) = %s' % self.hypothesis(i))
                # print("E_i: ", E_i, "\nalpha_i: ", alpha_i, "\ny_i: ", y_i)

                ############################################################
                # if y^(i) * E_i < -tolerance and alpha_i < C \
                # or \
                # y^(i) * E_i > tolerance and alpha_i > 0
                ############################################################

                if ((y_i * E_i < -self.tolerance and alpha_i < self.C) or (y_i * E_i > self.tolerance and alpha_i > 0)):

                    # Select j != i randomly.
                    j = i  # j is the index of the sample that will be changed
                    while j == i:
                        # get j from 0 to num_sample-1
                        j = np.random.randint(0, num_sample)

                    # print("i, j: ", i, j)

                    ############################################################
                    # Calculate Error[j] = f(x^(j)) - y^(j)
                    # Where f(x^(j)) is the hypothesis
                    # f(x) = alpha * y * k(x^(j), x) + b
                    ############################################################
                    y_j = self.label[j]
                    E_j = self.hypothesis(j) - y_j
                    # print("E_j: ", E_j, "E_i: ", E_i)

                    alpha_i_old = self.alpha[i].copy()
                    alpha_j_old = self.alpha[j].copy()

                    ############################################################
                    # H, L: upper and lower bounds for alpha_sample_j (L <= alpha_j <= H)
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

                    sign = y_i * y_j
                    if sign < 0:  # y_i != y_j
                        L = max(0, self.alpha[j] - self.alpha[i])
                        H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
                    else:         # y_i == y_j
                        L = max(0, self.alpha[i] + self.alpha[j] - self.C)
                        H = min(self.C, self.alpha[i] + self.alpha[j])

                    # If L = H then skip this pair
                    if L == H:
                        continue

                    ############################################################
                    # We use eta to calculate the new value of alpha_j
                    # Calculate eta = 2 * kij - kii - kjj
                    ############################################################
                    kij, kii, kjj = self.k[i, j], self.k[i, i], self.k[j, j]
                    eta = 2 * kij - kii - kjj

                    ############################################################
                    # if eta > 0, clip alpha_j to L and H
                    # XXX for eta == 0 we have overflow problem
                    # "RuntimeWarning: invalid value encountered in true_divide" will be raised
                    # need to handle it
                    ############################################################
                    if eta >= 0:
                        if self.verbose:
                            print("eta>=0")
                        continue

                    self.alpha[j] -= y_j * (E_i - E_j) / eta
                    if self.alpha[j] > H:
                        self.alpha[j] = H
                    elif self.alpha[j] < L:
                        self.alpha[j] = L

                    if (np.abs(self.alpha[j] - alpha_j_old) < self.epsilon):
                        if self.verbose:
                            print("alpha_j not moving enough")
                        continue

                    ############################################################
                    # update alpha_i
                    # alpha_i = alpha_i_old + y_i * y_j * (alpha_j_new - alpha_j_old)
                    ############################################################
                    self.alpha[i] += (y_i * y_j) * \
                        (alpha_j_old - self.alpha[j])

                    ############################################################
                    # b is the threshold of the SVM
                    # b1 =  -E_i - y_i * (alpha_i_new - alpha_i_old) * kii - y_j * (alpha_j_new - alpha_j_old) * kij - b
                    # b2 =  -E_j - y_i * (alpha_i_new - alpha_i_old) * kij - y_j * (alpha_j_new - alpha_j_old) * kjj - b
                    # Update b to reflect change in alpha_i, alpha_j and to take into account the new margins
                    ############################################################

                    b_one = -(E_i + y_i * (self.alpha[i] - alpha_i_old) * kii + y_j * (
                        self.alpha[j] - alpha_j_old) * kij) + self.b
                    b_two = -(E_j + y_i * (self.alpha[i] - alpha_i_old) * kij + y_j * (
                        self.alpha[j] - alpha_j_old) * kjj) + self.b

                    if (0 < self.alpha[i]) and (self.alpha[i] < self.C):
                        self.b = b_one
                        if self.verbose:
                            print("b = i ", self.b)
                    elif (0 < self.alpha[j]) and (self.alpha[j] < self.C):
                        self.b = b_two
                        if self.verbose:
                            print("b = j ", self.b)
                    else:
                        self.b = (b_one + b_two) / 2
                        if self.verbose:
                            print("b = else ", self.b)

                    y_i = self.label[i]
                    y_j = self.label[j]

                    alpha_i = self.alpha[i].copy()
                    alpha_j = self.alpha[j].copy()

                    self.weight = self.weight + y_i * \
                        (alpha_i - alpha_i_old) * \
                        self.data[i] + y_j * \
                        (alpha_j - alpha_j_old) * self.data[j]

                    num_changed_alphas += 1  # num_changed_alphas is the number of alphas that changed
                    if self.verbose:
                        print("pass = %d i: %d, %d pairs changed" %(passes, i, num_changed_alphas))
            ############################################################
            # If no alpha_i, alpha_j have been updated then terminate
            ############################################################
            if num_changed_alphas == 0:
                passes += 1  # increment passes
            else:
                passes = 0  # reset passes to 0
            if self.verbose:
                print("SVM training iteration = %d finished" % passes)
        # alphas are the coefficients of the support vectors
        # b is the threshold of the SVM
        # return self.lagrange_multiplier, self.b, self.weight
        else:
            if self.verbose:
                print("Training Finished")

    def fit(self):
        self.sequential_minimal_optimization()
        return self.alpha, self.b, self.weight

    # def f(self, data): # 
    #     # return (-self.weight[0][0] * data - self.b + self.C) / self.weight[0][1]
    #     return 

    def decision_function(self, data):
        return np.inner(data, self.weight) + self.b

    def sign(self, data):
        fx = self.decision_function(data) 
        positive = (1) * (fx >= 0)
        negative = (-1) * (fx < 0)
        self.y_predict = positive + negative
        return self.y_predict

    def predict(self, data):
        self.support_vectors_indices_ = self.get_support_vector_indices()
        self.support_vectors_ = self.get_support_vector_()
        return self.sign(data)

    def accuracy(self, y_true, x_test):
        return np.mean(y_true == self.sign(x_test))


    # def geo_margin(self, data):
    #     return self.label * (self.weight / np.linalg.norm(self.weight)).T @ data + (self.b / np.linalg.norm(self.weight)).flatten()

    # def functional_margin(self, data): 
    #     return self.label * (self.weight.T @ data + self.b).flatten()

    # def get_support_vectors_indices(self):
    #     return np.where(self.alpha > 0)[0]

    def num_support_vectors(self):
        return np.sum(self.alpha > 0)

    
    def get_support_vector_indices(self):
        self.support_vectors_indices_ = np.where(self.alpha > 0)[0]
        return self.support_vectors_indices_

    def get_support_vector_(self):
        self.support_vectors_ = self.data[self.get_support_vector_indices()]
        return self.support_vectors_

    def get_params(self):
        return {'kernel type': self.kernel, 'C': self.C, 'tolerance': self.tolerance, 'max_passes': self.max_passes, 'epsilon': self.epsilon, 'degree': self.p_degree, 'gamma': self.sigma, 'coeffs': [*self.weight.flatten()], 'bias': self.b}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def __repr__(self) -> str:
        represent = "::\n Kernel_type: {}\nC: {:.2f}\nTolerance: {:.2f}\nMax_passes: {}\nEpsilon: {:.2f} \n Weight".format(
            self.kernel, self.C, self.tolerance, self.max_passes, self.epsilon, self.b, self.weight)
        if self.kernel == 'polynomial':
            represent += "\tdegree: {}:".format(self.p_degree)
        elif self.kernel == 'rbf' or self.kernel == 'gauassian':
            represent += "\tsigma: {:.2f}".format(self.sigma)
        return represent

    def plot_decision_boundry_2d(self, model, plt_title, data, label, axes=None):
        if axes is None:
            plt.gca()
        else:
            plt.axes(axes)

        X = data.copy()
        y = label.copy()

        x_limit = [np.min(X[:, 0]), np.max(X[:, 0])]
        y_limit = [np.min(X[:, 1]), np.max(X[:, 1])]

        x_mesh, y_mesh = np.meshgrid(np.linspace(*x_limit, 300), np.linspace(*y_limit, 300))
        rgb = np.array([[210, 0, 0], [0, 0, 150]]) / 255.0  # rgb is the color of the points

        helper = model.decision_function(
            np.c_[x_mesh.ravel(), y_mesh.ravel()]).reshape(x_mesh.shape)

        z_model = helper.copy()

        # if self.verbose:
        print("----------------------------------------------------")
        print("\n".join(["{}: {}".format(k, v) for k, v in model.get_params().items()]))
        print("----------------------------------------------------")

        plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
        # levels are the levels of the contour lines
        plt.contourf(x_mesh, y_mesh, np.sign(z_model.reshape(x_mesh.shape)), alpha=0.3, levels=2, cmap=ListedColormap(rgb), zorder=1)
        plt.contour(x_mesh, y_mesh, z_model, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
        plt.title(plt_title)
        return helper

    def decision_boundary(self, model, plt_title, data, label, ax=None):
        if ax is None:
            ax = plt.gca()
        else:
            ax = plt.axes(ax)

        X = data.copy()
        y = label.copy()

        x_limit = np.arange(start = X[:, 0].min() , stop =X[:, 0].max() , step =0.01)
        y_limit = np.arange(start = X[:, 1].min() , stop =X[:, 1].max() , step =0.01)

        x_mesh, y_mesh = np.meshgrid(x_limit, y_limit)
        test = np.array([x_mesh.ravel(), y_mesh.ravel()]).T
        Z = model.predict(test).reshape(x_mesh.shape).astype(np.int64)
        # print(Z)
        
        # Z_scaled = model.support_vector_ - model.support_vector_.min(axis=0)
        # Z_scaled = Z_scaled / Z_scaled.max(axis=0)
        
        plt.scatter(X[y == 1, 0], X[y == 1, 1], color=ListedColormap(('r', 'b'))(0), marker='o', label='1')
        plt.scatter(X[y == -1, 0], X[y == -1, 1], color=ListedColormap(('r', 'b'))(1), marker='x', label='-1')
        plt.contourf(x_mesh, y_mesh, Z, alpha=0.3, levels=2, cmap=ListedColormap(('r', 'b')), zorder=1)
        plt.xlim(x_mesh.min(), x_mesh.max())
        plt.ylim(y_mesh.min(), y_mesh.max())

        plt.title(plt_title)
        plt.xlabel('x_mesh')
        plt.ylabel('y_mesh')
        plt.legend(loc='best')

if __name__ == '__main__':

    part_one    = True
    part_two    = False
    part_three  = False
    part_four   = False
    part_five   = False

    if part_one:
        # Todo:
        # Implement Custom_SVM class
        # Consider kernel = linear, C = [0.01, 0.1, 1, 10, 100], tolerance = 0.001, max_passes = 5, epsilon = 0.001
        # plot decision boundary for each C and get accuracy for each C
        # report the best C and the corresponding accuracy
        
        dataset = pd.read_csv('d1.csv', header=None, names=['x1', 'x2', 'y'])
        X = dataset.iloc[:, :-1]
        y = dataset.iloc[:, -1]
        dataset['y'] = dataset['y'].replace(0, -1)

        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=0.18)

        y_train = y_train.reshape(-1)
        y_test = y_test.reshape(-1)

        C_list = [0.01, 0.1, 1, 10, 100]
        tolerance = 0.001
        max_passes = 5
        epsilon = 0.001
        b = 0
        w = np.zeros((1, x_train.shape[1]))
        alpha = np.zeros((x_train.shape[0]))

        model_result = []
        for C in C_list:
            model = SVM_custom(x_train, y_train, kernel='linear', C=C, tolerance=tolerance, max_passes=max_passes, epsilon=epsilon, b=b, weight=w, alpha=alpha)
            model.fit()
            accuracy = model.accuracy(y_test, x_test)
            model_result.append([C, accuracy])
        
        model_result = pd.DataFrame(model_result, columns=['C', 'accuracy'])
        print(model_result)

        Optimal_C = model_result.loc[model_result['accuracy'].idxmax()]['C']
        custom_model = SVM_custom(x_train, y_train, kernel='linear', C=Optimal_C, tolerance=tolerance, max_passes=max_passes, epsilon=epsilon, b=b, weight=w, alpha=alpha)
        custom_model.fit()

        sklearn_model = SVC(kernel='linear', C=Optimal_C, tol=tolerance, max_iter=max_passes, verbose=False)
        sklearn_model.fit(x_train, y_train)


        # Todo: decision boundry for custom model is not working
        # fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(18, 8))
        # custom_model.plot_decision_boundry_2d(data=np.array(X), label=np.array(y), model=custom_model, axes=axs[0], plt_title="::Custom SVM::")
        # custom_model.plot_decision_boundry_2d(data=np.array(X), label=np.array(y), model=sklearn_model, axes=axs[1], plt_title="::SKLearn-SVM::")
        # plt.show()

        custom_model.get_support_vector_()

        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(18, 8))
        custom_model.decision_boundary(model=custom_model, plt_title="::Custom SVM::", data=np.array(X), label=np.array(y), ax=axs[0])
        custom_model.decision_boundary(model=sklearn_model, plt_title="::SKLearn-SVM::", data=np.array(X), label=np.array(y), ax=axs[1])
        plt.show()

    if part_two:
        # Todo:
        # Implement gaussian kernel
        # Verify the accuracy of implementation of gaussian kernel with below parameters
        # x_1 = [1,2,1], x_2 = [0,4,-1], sigma = 2 => 324562.0

        dataset = pd.read_csv('d2.csv', header=None)
        X = dataset.iloc[:, :-1]
        y = dataset.iloc[:, -1]
        model = SVM_custom(X, y, kernel="gaussian", weight=np.zeros((1, X.shape[1])), alpha=np.zeros((X.shape[0])))
        kernel_result = model.gaussian_kernel(np.array([1, 2, 1]), np.array([0, 4, -1]), sigma=2) # 0.32465246735834974
        print("gaussian kernel result: ", kernel_result)

    if part_three:
        # Todo:
        # Implement gaussian kernel on d2.csv
        # Consider C = 1, sigma = 0.1, tolerance = 0.001, max_passes = 5, epsilon = 0.001
        dataset = pd.read_csv('d2.csv', header=None, names=['x1', 'x2', 'y'])
        X = dataset.iloc[:, :-1]
        y = dataset.iloc[:, -1]
        dataset['y'] = dataset['y'].replace(0, -1)
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)
        y_train = y_train.reshape(-1)
        y_test = y_test.reshape(-1)

        C = 1
        b = 0
        sigma = 0.1
        max_passes = 5
        verbose = False
        epsilon = 1e-5
        tolerance = 0.001
        w = np.zeros((1, x_train.shape[1]))
        alpha = np.zeros((x_train.shape[0]))
        custom_model = SVM_custom(x_train, y_train, kernel='gaussian', C=C, tolerance=tolerance, max_passes=max_passes, epsilon=epsilon, b=b, weight=w, alpha=alpha, sigma=sigma, verbose=verbose)
        custom_model.fit()

        # calculate support_vector
        support_vector_indices = []
        for i in range(len(custom_model.alpha)):
            if custom_model.alpha[i] > 0:
                support_vector_indices.append(i)
        print("support_vector_indices: ", support_vector_indices)

        # calculate dual_coef
        dual_coef = []
        for i in range(len(custom_model.alpha)):
            if custom_model.alpha[i] > epsilon:
                dual_coef.append(custom_model.label[i] * custom_model.alpha[i])
        print("dual_coef: ", dual_coef)
        
        # calculate intercept of custom_model
        intercept = 0
        for i in range(len(custom_model.alpha)):
            if custom_model.alpha[i] > epsilon:
                intercept += custom_model.label[i] * custom_model.alpha[i] * custom_model.gaussian_kernel(custom_model.data[i], custom_model.data[support_vector_indices[0]], sigma=sigma)
        intercept -= custom_model.b
        print("intercept: ", intercept)

        support_vector_indices = np.array(support_vector_indices)
        dual_coef = np.array(dual_coef)

        # calculate decision_function
        # diff = support_vector - z_scaled
        # note: z_scaled = (x - mean) / std
        # diff = x_train[support_vector_indices] - x_train[support_vector_indices].mean(axis=0)
        # diff = diff / diff.std(axis=0)
        # norm2 = np.linalg.norm(diff, axis=1)
        # dec_func_vec = -1 * (dual_coef.dot(np.exp(-sigma* (norm2**2)))) - intercept
        # print("decision_function: ", dec_func_vec)


        sklearn_model = SVC(kernel='rbf', C=C, tol=tolerance, max_iter=max_passes, gamma=sigma, verbose=verbose)
        sklearn_model.fit(x_train, y_train)

        # plot decision boundary
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
        z = custom_model.plot_decision_boundry_2d(data=np.array(X), label=np.array(y), model=custom_model, axes=axs[0], plt_title="::Custom SVM::")
        z1 = custom_model.plot_decision_boundry_2d(data=np.array(X), label=np.array(y), model=sklearn_model, axes=axs[1], plt_title="::SKLearn-SVM::")
        plt.show()


        # save decision_function_result to txt file
        with open('./z.txt', 'w') as f:
            for item in z:
                f.write("%s\n" % item)
        with open('./z1.txt', 'w') as f:
            for item in z1:
                f.write("%s\n" % item)

    if part_four:
        # Todo:
        # Use SVM with gaussian kernel on d3.csv
        # C and sigma range is [30, 10, 3, 1, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001]
        # Then check the accuracy of each model with different C and sigma on d3-validation.csv
        # get optimal C and sigma of the model with highest accuracy

        dataset = pd.read_csv('d3.csv', header=None, names=['x1', 'x2', 'y'])
        X = dataset.iloc[:, :-1]
        y = dataset.iloc[:, -1]
        y = y.replace(0, -1)
        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, shuffle=True)

        y_train = y_train.reshape(-1)
        y_test = y_test.reshape(-1)

        validation_set = pd.read_csv('d3-validation.csv', header=None, names=['x1', 'x2', 'y'])
        X_validation = validation_set.iloc[:, :-1]
        y_validation = validation_set.iloc[:, -1]
        y_validation = y_validation.replace(0, -1)
        x_validation = X_validation.values
        y_validation = y_validation.ravel()

        b = 0
        w = np.zeros((1, x_train.shape[1]))
        alpha = np.zeros((x_train.shape[0]))    
        C_list = [30, 10, 3, 1, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001]
        sigma_list = [30, 20, 10, 3, 1, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001]

        model_result = []
        for C in C_list:
            for sigma in sigma_list:
                svm_gausssian = SVM_custom(data=x_train, label=y_train, kernel="gaussian", b=b, alpha=alpha,
                                           weight=w, C=C, tolerance=0.001, max_passes=100, epsilon=1e-5, sigma=sigma)
                svm_gausssian.fit()
                accuracy = svm_gausssian.accuracy(y_validation, X_validation)
                model_result.append([C, sigma, accuracy])
                print("model result: ", model_result[-1])

        # choose the model with highest accuracy
        model_result = np.array(model_result)
        model_result = model_result[model_result[:, 2].argsort()]
        print(model_result[-1])
        optimal_C, optimal_sigma = model_result[-1][0], model_result[-1][1]

    if part_five:
        # Todo:
        # Plot the decision boundary of the model with highest accuracy
        # Use SVM with gaussian kernel on d3.csv and d3-validation.csv

        dataset = pd.read_csv('d3.csv', header=None, names=['x1', 'x2', 'y'])

        X = dataset.iloc[:, :-1]
        y = dataset.iloc[:, -1]
        y = y.replace(0, -1)
        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, shuffle=True)

        y_train = y_train.reshape(-1)
        y_test = y_test.reshape(-1)

        df = pd.read_csv("d3-validation.csv", header=None, names=['x1', 'x2', 'y'])
        print(df.head().reset_index())
        # replace 0 to -1 in column y
        df['y'] = df['y'].replace(0, -1)

        x_validation = df.iloc[:, :-1]
        y_validation = df.iloc[:, -1]

        print(x_validation)

        print("x_train: ", x_validation.shape, "y_train: ", y_validation.shape)
        print("x_test: ", x_test.shape, "y_test: ", y_test.shape)
        print("x_validation: ", x_validation.shape, "y_validation: ", y_validation.shape)
        b = 0
        C = 0.001
        sigma = 0.001
        w = np.zeros((1, x_train.shape[1]))
        alpha = np.zeros((x_train.shape[0]))
        svm_gausssian = SVM_custom(data=x_train, label=y_train, kernel="gaussian", b=b, alpha=alpha, 
                                    weight=w, C=C, sigma=sigma, tolerance=0.001, max_passes=100, epsilon=1e-5)
        svm_gausssian.fit()
        
        # fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        # svm_gausssian.plot_decision_boundry_2d(data=np.array(X), label=np.array(y), model=svm_gausssian, plt_title="::Train Data::", axes=axs[0])
        # svm_gausssian.plot_decision_boundry_2d(data=np.array(x_validation), label=np.array(y_validation), model=svm_gausssian, plt_title="::Validation Data::", axes=axs[1])
        # plt.show()

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        svm_gausssian.decision_boundary(data=np.array(X), label=np.array(y), model=svm_gausssian, plt_title="::Train Data::", ax=axs[0])
        plt.show()