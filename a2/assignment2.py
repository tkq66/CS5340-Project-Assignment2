from cv2 import imread, imwrite
import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
from sys import argv
import uuid

k = 2  # Segment foreground and background


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
    # Black and white mask
    mask = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    maskInverted = np.array([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]])
    # Apply the mask of k onto the pixel
    maskedImage = mask[pixelsClassAssignment].reshape(inputData.shape)
    maskedImageInverted = maskInverted[pixelsClassAssignment].reshape(inputData.shape)
    return maskedImage, maskedImageInverted


def outputDicttoJson(fileName, model):
    with open(fileName, "w") as fp:
        json.dump(model, fp)


def outputSegmentation(image, maskTuple, fileName):
    filteTitle, extension = fileName.split(".")
    maskFileTitle = "output/" + filteTitle + "-mask-" + str(uuid.uuid4()) + "." + extension
    maskInvFileTitle = "output/" + filteTitle + "-mask-inv-" + str(uuid.uuid4()) + "." + extension
    segFileTitle = "output/" + filteTitle + "-seg-" + str(uuid.uuid4()) + "." + extension
    segInvFileTitle = "output/" + filteTitle + "-seg-inv-" + str(uuid.uuid4()) + "." + extension
    maskedImage, maskedImageInverted = maskTuple
    segmentedImage = np.multiply(image, maskedImage)
    segmentedImageInverted = np.multiply(image, maskedImageInverted)
    imwrite(maskFileTitle, maskedImage * 255)
    imwrite(maskInvFileTitle, maskedImageInverted * 255)
    imwrite(segFileTitle, segmentedImage)
    imwrite(segInvFileTitle, segmentedImageInverted)


def main():
    fileName = argv[1]

    # Train data, assuming convergence criteria is logLikelihood = 0
    oldLogLikelihood = 1000000
    logLikelihood = 1000000
    patience = 5
    patienceCounter = 0
    delta = 1e-6
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

        maskTuple = segmentImage(k, inputData, old_mean, old_cov)
        plt.imshow(maskTuple[0])
        plt.pause(0.1)
        plt.draw()

        # Check for convergence and output the model and the file
        diff = oldLogLikelihood - logLikelihood
        print(diff)
        if diff < delta:
            patienceCounter += 1
        else:
            patienceCounter = 0
            oldLogLikelihood = logLikelihood
        if patienceCounter > patience:
            modelObject = {"means": old_mean.tolist(), "cov": old_cov.tolist(), "mix": old_mix.tolist()}
            outputDicttoJson("model.json", modelObject)
            outputSegmentation(inputData, maskTuple, fileName)
            break

        responsibility = expectation(k, inputData, old_mean, old_cov, old_mix)
        mean, cov, mix = maximization(k, responsibility, inputData, old_mean)
        old_mean = mean
        old_cov = cov
        old_mix = mix
        i += 1


if __name__ == "__main__":
    main()
