from cv2 import imread
import numpy as np
from scipy.stats import multivariate_normal
from sys import argv


def initialize(k, inputData):
    height, width, channels = inputData.shape

    # Initialize the mean
    h = np.random.choice(np.arange(height), k, replace=False)
    w = np.random.choice(np.arange(width), k, replace=False)
    initMean = inputData[h, w, :]

    # Initialzie covariance
    initCov = np.random.uniform(0, 50, size=(k, channels, channels))
    for i, x in enumerate(initCov):
        x = x.T.dot(x)
        initCov[i] = x

    # Initialize mixing coefficient
    initMixCoeff = np.repeat((1 / k), k)

    # Calculate log likelihood for the init values
    initLogLikelihood = evaluateLogLikelihood(inputData, initMean, initCov, initMixCoeff, k)

    return initMean, initCov, initMixCoeff, initLogLikelihood


def expectation(x, mean, cov, mixCoeff):
    responsibility = None
    h, w, c = x.shape
    data = x.reshape(h * w, c)

    N = []
    for i in range(mean.shape[0]):
        N.append(multivariate_normal.pdf(data, mean=mean[i, :], cov=cov[i, :]))
    N = np.array(N)
    N = N.T
    numerator = mixCoeff * N
    denominator = numerator.sum(axis=1) # for each N add values of K
    denominator = denominator.reshape(1,-1) # convert to row vector
    responsibility = numerator / denominator.T # denominator.T is a column vector thus can numerator divide by column
    print(denominator.shape)
    return responsibility


def maximization(responsibility, x, means, ):
    new_cov = None
    new_mixcoeff = None
    N_k = responsibility.sum(axis=0) # for each K add values of N
    x = x.reshape(responsibility.shape[0],-1)
    print(x.shape)
    # CALCULATING NEW MEANS
    new_means = []

    for i in range(responsibility.shape[1]): # for each k
        numerator = 0
        for index,pix in enumerate(x):
            numerator+= responsibility[index,i]*pix
        denominator = N_k[i]
        value = numerator/denominator
        new_means.append(value)

    new_means = np.array(new_means)

    # CALCULATING NEW COVARIANCES
    new_cov = []
    for i in range(responsibility.shape[1]): # for each k
        numerator = 0
        for index,pix in enumerate(x):
            temp = np.array(pix-new_means[i])
            temp = temp.reshape(1,-1)
            temp = responsibility[index,i] * temp * temp.T
            numerator+= temp
        denominator = N_k[i]
        value = numerator/denominator
        new_cov.append(value)
    new_cov = np.array(new_cov)

    # CALCULATING NEW MIX COEF
    new_mixcoeff = []
    N = responsibility.shape[0]
    for i in N_k:
        new_mixcoeff.append(i/N)
    new_mixcoeff = np.array(new_mixcoeff)

    return new_means, new_cov, new_mixcoeff


def evaluateLogLikelihood(inputData, mean, cov, mixCoeff, k):
    height, width, channels = inputData.shape
    formattedInput = inputData.reshape(height * width, channels).astype(np.float64)

    auxInput = np.zeros(height * width)
    for i in range(k):
        auxInput += mixCoeff[i] * multivariate_normal.pdf(formattedInput, mean=mean[i], cov=cov[i])
    logLikelihood = np.sum(np.log(auxInput))

    return -logLikelihood


def main():
    k = int(argv[1])
    fileName = argv[2]

    inputData = imread(fileName)
    initMean, initCov, initMixCoeff, initLogLikelihood = initialize(k, inputData)
    print('init means=', initMean.shape)
    print('init cov=', initCov.shape)
    print('init mixcoef=', initMixCoeff.shape)
    responsibility = expectation(inputData, initMean, initCov, initMixCoeff)
    new_means, new_cov, new_mix = maximization(responsibility,inputData,initMean)
    print('new means=',new_means.shape)
    print('new cov=', new_cov.shape)
    print('new mixcoef=', new_mix.shape)

    print('old=\n',initMean)
    print('\nnew=\n',new_means)

if __name__ == "__main__":
    main()
