from matplotlib import colors
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import scipy as sp
import plotly.express as px

def train_test_split(x_data, y_data=None, train_size=None, test_size=None, random_state=None, shuffle=None) -> tuple:
    """split dataset into train, validation and test
    @param data: user input dataset
    @param train_size: user input train size
    @param test_size: user input test size
    @param random_state: user input random state
    @return: train, validation and test dataset
    """
    if train_size is None and test_size is None:
        raise ValueError("train_size and test_size can not be both None")
    if train_size is not None and test_size is not None and train_size + test_size > 1:
        raise ValueError("train_size and test_size sum must be equal to 1")
    if train_size is not None and test_size is not None and train_size + test_size < 1:
        raise ValueError("train_size and test_size sum is not equal to one")
    if train_size is not None:
        if train_size <= 0:
            raise ValueError("train_size must be greater than 0")
        if train_size >= 1:
            raise ValueError("train_size must be less than 1")
        if test_size is None:
            test_size = 1 - train_size
        elif test_size <= 0:
            raise ValueError("test_size must be greater than 0")
        elif test_size >= 1:
            raise ValueError("test_size must be less than 1")
    if test_size is not None:
        if test_size <= 0:
            raise ValueError("test_size must be greater than 0")
        if test_size >= 1:
            raise ValueError("test_size must be less than 1")
        if train_size is None:
            train_size = 1 - test_size
        elif train_size <= 0:
            raise ValueError("train_size must be greater than 0")
        elif train_size >= 1:
            raise ValueError("train_size must be less than 1")
    
    if y_data is not None:
        data = pd.concat([x_data, y_data], axis=1)
    if y_data is None:
        data = x_data.copy(deep=True)

    train_size_relative_to_dataset = int(len(data) * train_size)
    # test_size_relative_to_dataset = int(len(data) * test_size)

    if shuffle is True:
        data = data.sample(frac=1).reset_index(drop=True)
    
    train_index = np.random.choice(len(data), train_size_relative_to_dataset, replace=False)
    
    # split train data
    x_train = data.iloc[train_index, :-1].reset_index(drop=True)
    y_train = data.iloc[train_index, -1:].reset_index(drop=True)
    
    # differnetiate train indexes from whole data indexes
    test_index = np.setdiff1d(np.arange(len(data)), train_index)

    # split test data
    x_test = data.iloc[test_index, :-1].reset_index(drop=True)
    y_test = data.iloc[test_index, -1:].reset_index(drop=True)

    return np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)

# def linear(x, z):  # x is a row vector, z is a column vector
#     return np.dot(x, z.T)

# def polynomial(x_i, z_j, ploynomial_degree):  # ploynomial_degree is the degree of the polynomial
#     return (np.dot(x_i, z_j.T) + 1) ** ploynomial_degree

# # sigma is the standard deviation and sigma^2 is the variance
# def guassian(x_i, z_j, sigma):
#     return np.exp(-np.linalg.norm(x_i - z_j, axis=1)**2 / (2 * (sigma**2)))

class SVM:
    def __init__(self, C, tolerance, max_passes, kernel='linear', ploynomial_degree=None, sigma=None, epsilon=1e-5):
        self.data = None # training data
        self.label = None # training label
        self.kernel = kernel  # kernel function 'linear', 'polynomial', 'guassian'
        self.C = C  # C is the penalty parameter of the error term
        self.tolerance = tolerance  # tolerance is used to determine when to stop the algorithm (stop when the difference between the two consecutive passes is less than tolerance)
        self.max_passes = max_passes # max_passes is the maximum number of passes through the data
        self.b = 0  # threshold for solution
        self.alphas = None   # alphas are the Lagrange multipliers for solution
        self.ploynomial_degree = ploynomial_degree  # ploynomial_degree is the degree of the polynomial for polynomial kernel
        self.sigma = sigma # sigma is the standard deviation for guassian kernel
        self.epsilon = epsilon  # epsilon is the tolerance for the error term
    
    def error(self, index:int):
        return self.label[index] - self.decision_function(self.data[index])

    def linear(self, x_i, z_j):
        return np.dot(x_i, z_j.T)

    def polynomial(self, x_i, z_j, ploynomial_degree):
        return (np.dot(x_i, z_j.T) + 1) ** ploynomial_degree

    def gaussian(self, x_i, z_j, sigma):
        return np.exp(-np.linalg.norm(x_i - z_j)**2 / (2 * (sigma**2)))

    def kernel_matrix(self, x_data, z_data):
        """
        @param x_data: training data
        @param z_data: testing data
        @return: kernel matrix
        """
        if self.kernel == 'linear':
            return self.linear(x_data, z_data)
        elif self.kernel == 'polynomial':
            return self.polynomial(x_data, z_data, self.ploynomial_degree)
        elif self.kernel == 'guassian':
            return self.guassian(x_data, z_data, self.sigma)
        else:
            raise ValueError("kernel is not supported")

    def hypothesis(self, index:int):
        # print((self.alpha * self.label.T).dot(self.kernel_matrix(self.data[index], self.data)) + self.b)
        # return  (self.alpha * self.label.T).dot(self.kernel_matrix(self.data[index], self.data)) + self.b
        # print(np.sum(self.alpha * self.label * self.k[index, :]) + self.b)
        return np.sum(self.alpha * self.label * self.k[index, :]) + self.b

    def sequential_minimal_optimization(self):
        """" ### Sequential Minimal Optimization
        C: regularization parameter
        tolerance: tolerance for stopping criterion
        max_passes: maximum # of times to iterate over alpha without changing
        alpha: Lagrange multipliers
        b: threshold
        return lagrange_multipliers, bias
        """
        # initialize alpha, b and passes
        num_sample, num_feature = self.data.shape # num_sample is the number of samples
        self.alpha = np.zeros_like(self.label, dtype=np.float64)
        self.b = 0
        passes = 0 # passes is the number of passes through the data
        error = np.zeros(num_sample)

        # loop until the number of passes is greater than max_passes or the difference between the current error and the previous error is less than the tolerance
        while passes < self.max_passes:
            num_changed_alphas = 0 # count the number of alphas that changed

            for i in range(num_sample): # for each training sample

                ############################################################
                #### Calculate Error[i] = f(x^(i)) - y^(i) where f(x^(i)) is the hypothesis -> f(x) = sum(alpha_i * y^(i) * K(x^(i), x)) + b
                ############################################################
                error[i] = self.hypothesis(i) - self.label[i]

                ############################################################
                #### If label[i]*E_i < -tolerance and alpha_i < C or if label[i]*E_i > tolerance and alpha_i > 0
                ############################################################
                if ((self.label[i] * error[i] < -self.tolerance and self.alpha[i] < self.C) or (self.label[i] * error[i] > self.tolerance and self.alpha[i] > 0)):

                    # Select j != i randomly.
                    j = i # j is the index of the sample that will be changed
                    while j == i:
                        j = np.random.choice(num_sample)

                    ############################################################
                    ## Calculate Error[j] = f(x^(j)) - y^(j) where f(x^(j)) is the hypothesis -> f(x) = sum(alpha_i * y^(i) * K(x^(i), x)) + b
                    ############################################################
                    error[j] = self.hypothesis(j) - self.label[j]

                    # Store old alpha_i, alpha_j
                    alpha_i_old = self.alpha[i]
                    alpha_j_old = self.alpha[j]

                    ############################################################
                    ## H, L: upper and lower bounds for alpha_sample_j (L <= alpha_j <= H)
                    ############################################################
                    if self.label[i] != self.label[j]:
                        L = max(0, self.alpha[j] - self.alpha[i])
                        H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
                    if self.label[i] == self.label[j]:
                        L = max(0, self.alpha[i] + self.alpha[j] - self.C)
                        H = min(self.C, self.alpha[i] + self.alpha[j])
                    
                    # If L = H then skip this pair
                    if L == H:
                        continue

                    ############################################################
                    ## Calculate eta = 2 * K(x^(i), x^(j)) - K(x^(i), x^(i)) - K(x^(j), x^(j))
                    ## We use eta to calculate the new value of alpha_j
                    ## eta = 0 if eta is not a number or if eta is a small negative number
                    ############################################################
                    eta = 2 * self.k[i, j] - self.k[i, i] - self.k[j, j]
                    
                    # if eta >= 0 then skip this pair
                    if eta >= 0:
                        continue

                    ############################################################
                    ## Calculate alpha_j^(new) = alpha_j^(old) - y^(j) * (E_i - E_j) / eta
                    ############################################################
                    self.alpha[j] -= self.label[j] * (error[i] - error[j]) / eta

                    # if alpha_new^(j) > H or alpha_new^(j) < L then set alpha_j^(new) = L or H else keep alpha_j^(new)
                    if self.alpha[j] > H:
                        self.alpha[j] = H
                    elif self.alpha[j] < L:
                        self.alpha[j] = L

                    # if alpha_new^(j) is within the bounds then skip to the next pair
                    if abs(self.alpha[j] - alpha_j_old) < self.epsilon:
                        continue

                    ############################################################
                    ## Calculate alpha_new^(i) = alpha_i^(old) + y^(i) * y^(j) * (alpha_j^(old) - alpha_j^(new))
                    ############################################################
                    self.alpha[i] += self.label[i] * self.label[j] * (alpha_j_old - self.alpha[j])

                    ############################################################
                    ## Update b to reflect change in alpha_i, alpha_j and to take into account the new margins
                    ## b is the threshold of the SVM 
                    ############################################################
                    b1 = self.b - error[i] - self.label[i] * (self.alpha[i] - alpha_i_old) * self.k[i, i] - self.label[j] * (self.alpha[j] - alpha_j_old) * self.k[i, j]
                    b2 = self.b - error[j] - self.label[i] * (self.alpha[i] - alpha_i_old) * self.k[i, j] - self.label[j] * (self.alpha[j] - alpha_j_old) * self.k[j, j]

                    ############################################################
                    ## Update b based on which one is closer to 0
                    ############################################################
                    if 0 < self.alpha[i] < self.C:
                        self.b = b1
                    elif 0 < self.alpha[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2

                    ## Update num_changed_alphas
                    num_changed_alphas += 1 # num_changed_alphas is the number of alphas that changed

            ############################################################
            ## If no alpha_i, alpha_j have been updated then terminate
            ############################################################
            if num_changed_alphas == 0:
                passes += 1 # increment passes
            else:
                passes = 0 # reset passes to 0

        ## alphas are the coefficients of the support vectors
        ## b is the threshold of the SVM
        return self.alpha, self.b

    def fit(self, data, label):
        self.data = data
        self.label = label
        self.k = self.kernel_matrix(self.data, self.data)
        self.alpha, self.b = self.sequential_minimal_optimization()
        return self

    def decision_function(self, data): # data is a matrix of samples to be classified
        # self.k = self.kernel_matrix(data, data).reshape(-1, 1) # 
        


    def predict(self, x_test):

        return np.sign(self.decision_function(x_test)) # sign function returns 1 if y > 0 else -1

    def score(self, y_test, y_predict):
        return np.mean(y_test == y_predict) # return the accuracy of the model

    def get_params(self):
        return {'kernel type': self.kernel, 'C': self.C, 'tolerance': self.tolerance, 'max_passes': self.max_passes, 'epsilon': self.epsilon}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def __repr__(self) -> str: 
        return 'SVM(kernel_type={}, C={}, tolerance={}, max_passes={}, epsilon={})'.format(self.kernel, self.C, self.tolerance, self.max_passes, self.epsilon)

    # def plot_decision_boundary(self, x_test, y_test):
    #     x_min, x_max = x_test[:, 0].min() - 1, x_test[:, 0].max() + 1 # x_min and x_max are the minimum and maximum values of the x-axis
    #     y_min, y_max = x_test[:, 1].min() - 1, x_test[:, 1].max() + 1 # y_min and y_max are the minimum and maximum values of the y-axis
    #     xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1)) # create a grid over the domain of the data
    #     Z = self.predict(np.c_[xx.ravel(), yy.ravel()]) # get the predicted class for each example in the grid
    #     Z = Z.reshape(xx.shape)
    #     plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
    #     plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, cmap=plt.cm.Paired)
    #     plt.xlabel('X')
    #     plt.ylabel('Y')
    #     plt.xlim(xx.min(), xx.max())
    #     plt.ylim(yy.min(), yy.max())
    #     plt.show()

    # def plot_decision_boundry(self, x_data, label_data):
    #     x_min, x_max = x_data[:, 0].min() - 1, x_data[:, 0].max() + 1
    #     y_min, y_max = x_data[:, 1].min() - 1, x_data[:, 1].max() + 1
    #     xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
    #                         np.arange(y_min, y_max, 0.1))
    #     grid_points = np.c_[xx.ravel(), yy.ravel()]
    #     grid_labels = self.predict(grid_points)
    #     grid_labels = grid_labels.reshape(xx.shape)
    #     plt.contourf(xx, yy, grid_labels, cmap=plt.cm.Paired, alpha=0.8)
    #     plt.scatter(x_data[:, 0], x_data[:, 1],
    #                 c=label_data, cmap=plt.cm.Paired)
    #     plt.xlabel('X1')
    #     plt.ylabel('X2')
    #     plt.show()

    # def plot_decision_boundry_2d(self, y_pred, axes):
    #     plt.axes(axes)
    #     x_limit = [np.min(self.data[:, 0]), np.max(self.data[:, 0])] # x_limit is the minimum and maximum values of the x-axis
    #     y_limit = [np.min(self.data[:, 1]), np.max(self.data[:, 1])] # y_limit is the minimum and maximum values of the y-axis
    #     x_mesh, y_mesh = np.meshgrid(np.linspace(x_limit[0], x_limit[1], 100),
    #                                  np.linspace(y_limit[0], y_limit[1], 100))
    #     # color for points
    #     color = np.array([[200, 0, 0], [0, 0, 100]]) / 255
    #     z_model = self.predict(np.c_[x_mesh.ravel(), y_mesh.ravel()]).reshape(x_mesh.shape)
    #     plt.scatter(self.data[:, 0], self.data[:, 1], c=self.label, cmap='summer')
    #     plt.contour(x_mesh, y_mesh, z_model, colors=color, levels=[-1, 0, 1], alpha=0.5, linestyle=['--', '-', '--'])
    #     plt.contourf(x_mesh, y_mesh, np.sign(z_model.reshape(x_mesh.shape)), alpha=0.2, cmap=ListedColormap(['#FF0000', '#0000FF']), zorder=1)
    #     plt.xlim(x_limit)
    #     plt.ylim(y_limit)
    #     plt.xlabel('X_1')
    #     plt.ylabel('X_2')
    #     plt.show()

    def plot_decision_boundry_2d(self,y_predict, axes):
        plt.axes(axes) 
        xlim = [np.min(self.data[:, 0]), np.max(self.data[:, 0])] # xlim is the x-axis limits
        ylim = [np.min(self.data[:, 1]), np.max(self.data[:, 1])] # ylim is the y-axis limits 
        xx, yy = np.meshgrid(np.linspace(*xlim, num=50), np.linspace(*ylim, num=50)) # xx and yy are the meshgrid of the x and y axis limits
        rgb = np.array([[210, 0, 0], [0, 0, 150]])/255.0 # rgb is the color of the points

        z_model = self.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape) # z_model is the decision function of the model i.e. the value of the hyperplane and ravel

        plt.scatter(self.data[:, 0], self.data[:, 1], c=self.label, s=50, cmap='autumn') # X[:, 0] is the first column of X, X[:, 1] is the second column of X and c=y means that the color of the points is the same as the class of the points
        plt.contour(xx, yy, z_model, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--']) # levels are the levels of the contour lines
        plt.contourf(xx, yy, np.sign(z_model.reshape(xx.shape)), alpha=0.3, levels=2, cmap=ListedColormap(rgb), zorder=1)
        plt.show()


# print(type(x_train))
# print(type(y_train))
# print(np.c_[x_train.shape, y_train.shape])
# print(np.c_[x_test.shape, y_test.shape])
# print(np.c_[x_train[:10], y_train[:10]])

dataset = pd.read_csv('d1.csv', header=None)
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

# y = y.replace(0, -1)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

svm_linear = SVM(kernel='linear', C=10, tolerance=0.001, max_passes=100, epsilon=0.000005)
svm_linear.fit(x_train, y_train)
y_predict = svm_linear.predict(x_test)
print(svm_linear.score(y_test, y_predict))
# svm_linear.plot_decision_boundry(x_test, y_predict)

sns.set()

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
svm_linear.plot_decision_boundry_2d(y_predict, axs[0])
svm_linear.plot_decision_boundry_2d(y_predict, axs[1])

class_A = 1
class_B = -1

def confusion_matrix(y_test, y_predict):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in range(len(y_test)):
        if y_test[i] == class_A and y_predict[i] == class_A:
            TP += 1
        elif y_test[i] == class_A and y_predict[i] == class_B:
            FN += 1
        elif y_test[i] == class_B and y_predict[i] == class_A:
            FP += 1
        elif y_test[i] == class_B and y_predict[i] == class_B:
            TN += 1
    # return TP, FP, TN, FN as a matrix
    return np.array([[TP, FP], [FN, TN]])

def plot_confusion(y_test, y_predict):
    mat = confusion_matrix(y_test, y_predict)
    plt.figure(figsize=(10, 10))
    sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
                xticklabels=['Predicted Class 0', 'Predicted Class 1'], yticklabels=['Actual Class 0', 'Actual Class 1'])
    plt.xlabel('true label')
    plt.ylabel('predicted label')
    plt.show()

# print('Custom SVM:')
# plot_confusion(y_test, y_predict)


print(np.c_[y_test[:10], y_predict[:10]])