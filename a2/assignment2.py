from cv2 import imread
import numpy as np
from scipy.stats import multivariate_normal
from sys import argv


def initializeMean(k, height, width, inputData):
    # Initialize the mean
    h = np.random.choice(np.arange(height), k, replace=False)
    w = np.random.choice(np.arange(width), k, replace=False)
    return inputData[h, w, :]


def initializeCov(k, channels, spread=50):
    # Initialzie covariance
    initCov = np.random.uniform(0, spread, size=(k, channels, channels))
    for i, x in enumerate(initCov):
        x = x.T.dot(x)
        initCov[i] = x
    return initCov


def initializeMix(k):
    # Initialize mixing coefficient
    return np.repeat((1 / k), k)


def initialize(k, inputData):
    height, width, channels = inputData.shape
    initMean = initializeMean(k, height, width, inputData)
    initCov = initializeCov(k, channels)
    initMixCoeff = initializeMix(k)
    return initMean, initCov, initMixCoeff


def expectation(k, x, mean, cov, mixCoeff):
    responsibility = None
    h, w, c = x.shape
    totalPixels = h * w
    data = x.reshape(totalPixels, c)

    numerator = np.empty((k, totalPixels))
    for i in range(k):
        numerator[i] = mixCoeff[i] * multivariate_normal.pdf(data, mean=mean[i], cov=cov[i])
    denominator = numerator.sum(axis=0)  # for each N add values of K
    responsibility = numerator / denominator  # denominator.T is a column vector thus can numerator divide by column
    return responsibility


def maximization(k, responsibility, x, means):
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


def evaluateLogLikelihood(k, inputData, mean, cov, mixCoeff):
    height, width, channels = inputData.shape
    formattedInput = inputData.reshape(height * width, channels).astype(np.float64)

    auxInput = np.zeros(height * width)
    for i in range(k):
        auxInput += mixCoeff[i] * multivariate_normal.pdf(formattedInput, mean=mean[i], cov=cov[i])
    logLikelihood = np.sum(np.log(auxInput))

    return -logLikelihood


def segmentImage(k, inputData, mean, cov):
    h, w, c = inputData.shape
    totalPixels = h * w
    formattedInput = inputData.reshape(totalPixels, c).astype(np.float64)

    # For each pixel, calculate probability for each k
    rawKImageClassProb = np.empty((k, totalPixels))
    for i in range(k):
        rawKImageClassProb[i] = multivariate_normal.pdf(formattedInput, mean=mean[i], cov=cov[i])
    # Choose the k given the max probability
    pixelsClassAssignment = np.argmax(rawKImageClassProb, axis=0)
    # Apply the mean of the k onto the pixel
    maskedImage = mean[pixelsClassAssignment].reshape(inputData.shape)

    return maskedImage


def main():
    k = int(argv[1])
    fileName = argv[2]

    # Train data, assuming convergence criteria is logLikelihood = 0
    logLikelihood = 1000000
    inputData = imread(fileName)
    height, width, channels = inputData.shape
    old_mean, old_cov, old_mix = initialize(k, inputData)
    i = 0
    while logLikelihood != 0:
        try:
            logLikelihood = evaluateLogLikelihood(k, inputData, old_mean, old_cov, old_mix)
        except np.linalg.LinAlgError:
            print("New covariance matrix is singular. Resetting...")
            old_cov = initializeCov(k, channels)
            continue
        print(i, logLikelihood)
        responsibility = expectation(k, inputData, old_mean, old_cov, old_mix)
        mean, cov, mix = maximization(k, responsibility, inputData, old_mean)
        old_mean = mean
        old_cov = cov
        old_mix = mix
        i += 1


if __name__ == "__main__":
    main()
