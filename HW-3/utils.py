import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


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


def create_dataset(N, D=2, K=2):
    X = np.zeros((N * K, D))  # data matrix (each row = single example)
    y = np.zeros(N * K)  # class labels

    for j in range(K):
        ix = range(N * j, N * (j + 1))
        r = np.linspace(0.0, 1, N)  # radius
        t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * 0.2  # theta
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = j

    # lets visualize the data:
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    plt.show()

    y[y == 0] -= 1

    return X, y


def plot_contour(X, y, svm):
    # plot the resulting classifier
    h = 0.01
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    points = np.c_[xx.ravel(), yy.ravel()]

    Z = svm.predict(points)
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)

    # plt the points
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    plt.show()


def k_fold(x_data, y_data, k_fold_number, shuffle=True, random_state=None):
    if random_state != None:
        np.random.seed(random_state)

    dataset = pd.concat([x_data, y_data], axis=1)
    if shuffle == True:
        dataset = dataset.sample(frac=1).reset_index(drop=True)
    dataset_size = len(dataset)
    fold_size = int(dataset_size / k_fold_number)
    indices = np.arange(dataset_size)
    if k_fold_number <= 1:
        raise ValueError("K-fold cross validation requires at least 2 folds")
    else:
        for index in range(k_fold_number):
            test_indices = indices[index * fold_size : (index + 1) * fold_size]
            train_indices =  np.delete(indices, test_indices)
            x_train = dataset.iloc[train_indices, :-1].reset_index(drop=True)
            y_train = dataset.iloc[train_indices, -1].reset_index(drop=True)
            x_test = dataset.iloc[test_indices, :-1].reset_index(drop=True)
            y_test = dataset.iloc[test_indices, -1].reset_index(drop=True)
            yield x_train, y_train, x_test, y_test

def cross_validation(x_data, y_data, k_fold_number, scoring, initial_theta, shuffle=True, alpha=0.1, lambda_param=1, verbose=True):
    """ ### Cross validation for logistic regression
    """

    if initial_theta is None:
        initial_theta = np.random.rand(x_data.shape[1]) * 10 + 5 # random values between 5 and 10

    k_fold_data = k_fold(x_data, y_data, k_fold_number, shuffle)

    metric_df = pd.DataFrame(columns=['alpha', 'lambda_param', 'accuracy', 'precision', 'recall', 'f1_score', 'loss', 'iteration'])
    average_metric_df = pd.DataFrame(columns=['alpha', 'lambda_param', 'accuracy', 'precision', 'recall', 'f1_score', 'loss', 'iteration'])

    # iterate through the data
    for x_train, y_train, x_test, y_test in k_fold_data:

        x_train, y_train, x_test, y_test = np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)
        y_train, y_test = y_train.ravel(), y_test.ravel()

        # train model
        coeffs, theta_history, train_loss_history, iteration = fit_logistic_regression(x_train, y_train, initial_theta , alpha=alpha, regularization_parameter=1, verbose=verbose)
        last_loss = train_loss_history[-1]
        y_predict = predict_logistic_regression(x_test, coeffs)

        # calculate metrics
        accuracy, precision, recall, f1_score, cm_result = calculate_metrics(y_test, y_predict)
        metric_df = metric_df.append({
            'alpha': alpha,
            'lambda_param': lambda_param,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'loss': last_loss,
            'iteration': iteration,
            'coeffs': coeffs
        }, ignore_index=True)

    iteration_mean = metric_df['iteration'].mean()
    min_loss_index = metric_df.loc[metric_df['loss'].idxmin()]
    coeffs = min_loss_index['coeffs']

    metric_df = metric_df.drop(columns=['coeffs'])
    metric_df = metric_df.drop(columns=['iteration'])

    average_metric_df = metric_df.groupby(['alpha', 'lambda_param']).mean().reset_index()
    return average_metric_df, coeffs, iteration_mean



def confusion_matrix(y_true, y_pred):
    c_matrix = np.zeros((2, 2)) # 2x2 matrix as problem is 2-class
    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred[i] == 1: # TP
            c_matrix[0][0] += 1
        elif y_true[i] == 0 and y_pred[i] == 1: # FP
            c_matrix[0][1] += 1
        elif y_true[i] == 1 and y_pred[i] == 0: # FN
            c_matrix[1][0] += 1
        elif y_true[i] == 0 and y_pred[i] == 0: # TN
            c_matrix[1][1] += 1
        TP, FP, FN, TN = c_matrix[0][0], c_matrix[0][1], c_matrix[1][0], c_matrix[1][1]
    # return TP, FP, FN, TN
    return c_matrix

def classification_report(y_true, y_pred):
    c_matrix = confusion_matrix(y_true, y_pred)
    TP, FP, FN, TN = c_matrix[0][0], c_matrix[0][1], c_matrix[1][0], c_matrix[1][1]
    accuracy = (TP + TN) / (TP + FP + FN + TN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * precision * recall / (precision + recall)
    return accuracy, precision, recall, f1_score

def show_metrics(confusion_result):
    TP, FP, FN, TN = confusion_result
    print("\nConfusion Matrix:")
    print("\t\tPredicted")
    print("\t\t\t0\t1")
    print("Actual")
    print("\t0\t\t{}\t{}".format(TP, FP))
    print("\t1\t\t{}\t{}".format(FN, TN))
    print(">===================================================<")
    print("Accuracy: ", np.round((TP + TN) / (TP + TN + FP + FN), 3))
    print("Precision: ", np.round(TP / (TP + FP), 3))
    print("Recall: ", np.round(TP / (TP + FN), 3))
    print("F1 Score: ", np.round(2 * TP / (2 * TP + FP + FN), 3))
    print(">===================================================<")


def calculate_accuracy(confusion_matrix_result):
    TP, FP, FN, TN = confusion_matrix_result
    return np.round((TP + TN) / (TP + TN + FP + FN), 5)

def calculate_precision(confusion_matrix_result):
    TP, FP, FN, TN = confusion_matrix_result
    return np.round(TP / (TP + FP), 5)

def calculate_recall(confusion_matrix_result):
    TP, FP, FN, TN = confusion_matrix_result
    return np.round(TP / (TP + FN), 5)

def calculate_f_beta_score(confusion_matrix_result, beta=1):
    """
    ### Calculates the F-β score (the weighted harmonic mean of precision and recall).
    ### A perfect model has an F-score of 1.
    """
    TP, FP, FN, TN = confusion_matrix_result
    return np.round( (1 + beta**2) * TP / ((1 + beta**2) * TP + (beta**2) * FP + FN), 5)


def plot_confusion(y_test, y_predict):
    mat = confusion_matrix(y_test, y_predict)
    plt.figure(figsize=(10, 10))
    sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
                xticklabels=['Predicted Class 0', 'Predicted Class 1'], yticklabels=['Actual Class 0', 'Actual Class 1'])
    plt.xlabel('true label')
    plt.ylabel('predicted label')
    plt.show()