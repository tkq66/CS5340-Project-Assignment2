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


def expectation(k, x, mean, cov, mixCoeff):
    responsibility = None
    h, w, c = x.shape
    totalPixels = h * w
    data = x.reshape(totalPixels, c)

    numerator = np.empty((k, totalPixels))
    for i in range(k):
        numerator[i] = mixCoeff[i] * multivariate_normal.pdf(data, mean=mean[i, :], cov=cov[i, :])
    denominator = numerator.sum(axis=0)  # for each N add values of K
    responsibility = numerator / denominator  # denominator.T is a column vector thus can numerator divide by column
    return responsibility


def maximization(k, responsibility, x, means, ):
    h, w, c = x.shape
    totalPixels = h * w
    data = x.reshape(totalPixels, c)

    N_k = responsibility.sum(axis=1)  # for each K add values of N

    # CALCULATING NEW MEANS
    tempMean = np.empty((k, totalPixels, c))
    for i in range(k):
        tempMean[i] = (data.T * responsibility[i]).T
    new_means = (tempMean.sum(axis=1).T / N_k).T

    # CALCULATING NEW COVARIANCES
    new_cov = np.empty((k, c, c))
    for i in range(k):
        difference = data - new_means[i]
        mod = (responsibility[i] * difference.T) / N_k[i]
        new_cov[i] = mod.dot(difference)

    # CALCULATING NEW MIX COEF
    new_mixCoeff = N_k / totalPixels

    return new_means, new_cov, new_mixCoeff


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
    responsibility = expectation(k, inputData, initMean, initCov, initMixCoeff)
    new_means, new_cov, new_mix = maximization(k, responsibility, inputData, initMean)


if __name__ == "__main__":
    main()
