import numpy as np
import scipy.linalg as sp_la
import pandas as pd
import math

def getSummaryStatistics(data):
    "Get the max, min, mean, var for each variable in the data."
    return pd.DataFrame(np.array([data.max(axis=0), data.min(axis=0), data.mean(axis=0), data.var(axis=0)]))

def getShapeType(data):
    "Get the shape and type of the data."
    return (data.shape, data.dtype)
    
def minmaxGlobal(data):
    "Global max-min normalization."
    scaleTransform = np.eye(data.shape[1], data.shape[1])
    for i in range(data.shape[1]):
        scaleTransform[i, i] = 1 / (data.max() - data.min())
    return (scaleTransform@data.T).T

def minmaxLocal(data):
    "Local max-min normalization."
    scaleTransform = np.eye(data.shape[1], data.shape[1])
    for i in range(data.shape[1]):
        if data[:, i].max() - data[:, i].min() != 0:
            scaleTransform[i, i] = 1 / (data[:, i].max() - data[:, i].min())
    return (scaleTransform@data.T).T

def zScore(data):
    "z score."
    homogenizedData = np.append(data, np.array([np.ones(data.shape[0], dtype=int)]).T, axis=1)
    translateTransform = np.eye(homogenizedData.shape[1])
    for i in range(homogenizedData.shape[1]):
        translateTransform[i, homogenizedData.shape[1]-1] = -homogenizedData[:, i].mean()
    diagonal = [1 / homogenizedData[:, i].std() if homogenizedData[:, i].std() != 0 else 1 for i in range(homogenizedData.shape[1])]
    scaleTransform = np.eye(homogenizedData.shape[1]) * diagonal
    data = (scaleTransform@translateTransform@homogenizedData.T).T
    return data[:, :data.shape[1]-1]

def fitlstsq(data, independent, dependent):
    "Fit a linear regression using least squares. Independent should be an array of indices of independent variables. Dependent should be one independent variable."
    # These are our independent variable(s)
    x = data[np.ix_(np.arange(data.shape[0]), independent)]
    # We add a column of 1s for the intercept
    A = np.hstack((np.array([np.ones(x.shape[0])]).T, x))
    # This is the dependent variable 
    y = data[:, dependent]
    # This is the regression coefficients that were fit, plus some other results
    c, res, _, _ = sp_la.lstsq(A, y)
    return c

def fitnorm(data, independent, dependent):
    "Fit a linear regression using the normal equation. Independent should be an array of indices of independent variables. Dependent should be one independent variable."
    # These are our independent variable(s)
    x = data[np.ix_(np.arange(data.shape[0]), independent)]
    # We add a column of 1s for the intercept
    A = np.hstack((np.array([np.ones(x.shape[0])]).T, x))
    # This is the dependent variable 
    y = data[:, dependent]
    # This is the regression coefficients that were fit, plus some other results
    c = sp_la.inv(A.T.dot(A)).dot(A.T).dot(y)
    return c

def fitqr(data, independent, dependent):
    "Fit a linear regression using QR decomposition. Independent should be an array of indices of independent variables. Dependent should be one independent variable."
    # These are our independent variable(s)
    x = data[np.ix_(np.arange(data.shape[0]), independent)]
    # We add a column of 1s for the intercept
    A = np.hstack((np.array([np.ones(x.shape[0])]).T, x))
    # This is the dependent variable 
    y = data[:, dependent]
    # This is the regression coefficients that were fit, plus some other results
    Q, R = sp_la.qr(A)
    print(A.shape)
    print(Q.shape)
    print(R.shape)
    c = sp_la.solve_triangular(R, Q.T.dot(y))
    return c
    
def gradient_descent(data, independent, dependent, lr, epochs):
    "Fit a linear regression using gradient descent. Independent should be an array of indices of independent variables. Dependent should be one independent variable."
    # initialize m and b
    rng = default_rng()
    c = rng.standard_normal(2)
    # set n, x and y for readability of the method
    n = data.shape[0]
    x = data[np.ix_(np.arange(data.shape[0]), independent)]
    y = data[:, dependent]
    A = np.hstack((np.array([np.ones(x.shape[0])]).T, x))
    for i in range(epochs):
        yhat = np.dot(A, c)
        # how are we doing on MSSE?
        print((1/n) * np.sum(y - yhat)**2)
        if ((1/n) * np.sum(y - yhat)**2) < 0.00001:
            return c
        # fill in the partial derivatives
        dpdm = (-2/n) * np.sum(x * (y - yhat))
        dpdb = (-2/n) * np.sum(y - yhat)
        # update c
        c = c - np.array([lr * dpdm, lr * dpdb])
    return c
    
def msse(y, yhat):
    "Calculate MSSE."
    if len(y) != len(yhat):
        print("Need y and yhat to be the same length!")
        return 0
    return (1 / len(y)) * (((y - yhat())**2).sum())

def rsquared(y, yhat):
    "Calculate R^2."
    if len(y) != len(yhat):
        print("Need y and yhat to be the same length!")
        return 0
    return 1 - (((y - yhat)**2).sum() / ((y - y.mean())**2).sum())

def predict(data, independent, c):
    "Given a linear regression function, predict on new data."
    # These are our independent variable(s)
    x = data[np.ix_(np.arange(data.shape[0]), independent)]
    # We add a column of 1s for the intercept
    A = np.hstack((np.array([np.ones(x.shape[0])]).T, x))
    return np.dot(A, c)