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
    denominator = numerator.sum(axis=0)
    responsibility = numerator / denominator

    return responsibility


def maximization():
    maximization = None
    return maximization


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
    responsibility = expectation(inputData, initMean, initCov, initMixCoeff)


if __name__ == "__main__":
    main()
