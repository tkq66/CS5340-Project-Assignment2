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
    initCov = np.random.uniform(0, 3, size=(k, channels, channels))
    for i, x in enumerate(initCov):
        x = x.T.dot(x)
        initCov[i] = x

    # Initialize mixing coefficient
    initMixCoeff = np.repeat((1 / k), k)

    # Calculate log likelihood for the init values
    initLogLikelihood = evaluateLogLikelihood(inputData, initMean, initCov, initMixCoeff)

    return initMean, initCov, initMixCoeff, initLogLikelihood


def expectation(mean, cov, mixCoeff):
    responsibility = None

    return responsibility


def maximization():
    pass


def evaluateLogLikelihood(inputData, mean, cov, mixCoeff):
    return


def main():
    k = int(argv[1])
    fileName = argv[2]

    inputData = imread(fileName)
    initMean, initCov, initMixCoeff, initLogLikelihood = initialize(k, inputData)
    print("mean", initMean)
    print("cov", initCov)


if __name__ == "__main__":
    main()
