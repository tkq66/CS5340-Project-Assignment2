import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal


class EM:

    def __init__(self):
        pass

    def initializeMean(self, k, height, width, inputData):
        # Initialize the mean
        h = np.random.choice(np.arange(height), k, replace=False)
        w = np.random.choice(np.arange(width), k, replace=False)
        return inputData[h, w, :]

    def initializeCov(self, k, channels, spread=50):
        # Initialzie covariance
        initCov = np.random.uniform(0, spread, size=(k, channels, channels))
        for i, x in enumerate(initCov):
            x = x.T.dot(x)
            initCov[i] = x
        return initCov

    def initializeMix(self, k):
        # Initialize mixing coefficient
        return np.repeat((1 / k), k)

    def initialize(self, k, inputData):
        height, width, channels = inputData.shape
        initMean = self.initializeMean(k, height, width, inputData)
        initCov = self.initializeCov(k, channels)
        initMixCoeff = self.initializeMix(k)
        return initMean, initCov, initMixCoeff

    def expectation(self, k, x, mean, cov, mixCoeff):
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

    def maximization(self, k, responsibility, x, means):
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

    def evaluateLogLikelihood(self, k, inputData, mean, cov, mixCoeff):
        height, width, channels = inputData.shape
        formattedInput = inputData.reshape(height * width, channels).astype(np.float64)

        auxInput = np.zeros(height * width)
        for i in range(k):
            auxInput += mixCoeff[i] * multivariate_normal.pdf(formattedInput, mean=mean[i], cov=cov[i])
        logLikelihood = np.sum(np.log(auxInput))

        return -logLikelihood

    def segmentImage(self, k, inputData, mean, cov):
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
        segmentedImage = np.multiply(inputData, maskedImage)
        segmentedImageInverted = np.multiply(inputData, maskedImageInverted)
        return maskedImage * 255, maskedImageInverted * 255, segmentedImage, segmentedImageInverted

    def run(self, k, inputData, initial_params=None, patience=5, delta=1e-6, verbose=False):
        assert k > 0
        assert isinstance(inputData, np.ndarray)
        assert len(inputData.shape) == 3
        assert (initial_params is None) or (initial_params is not None and len(initial_params) == 3)

        oldLogLikelihood = 1000000
        logLikelihood = 1000000
        patienceCounter = 0

        # Train data, assuming convergen4ce criteria is logLikelihood = 0
        height, width, channels = inputData.shape
        old_mean, old_cov, old_mix = self.initialize(k, inputData) if initial_params is None else initial_params
        while logLikelihood != 0:
            try:
                logLikelihood = self.evaluateLogLikelihood(k, inputData, old_mean, old_cov, old_mix)
            except np.linalg.LinAlgError:
                if verbose:
                    print("New covariance matrix is singular. Resetting...")
                old_cov = self.initializeCov(k, channels)
                continue

            output = self.segmentImage(k, inputData, old_mean, old_cov)

            # Check for convergence and output the model and the file
            diff = oldLogLikelihood - logLikelihood
            if verbose:
                plt.imshow(output[0] / 255)
                plt.pause(0.1)
                plt.draw()
                print(diff)
            if diff < delta:
                patienceCounter += 1
            else:
                patienceCounter = 0
                oldLogLikelihood = logLikelihood
            if patienceCounter > patience:
                modelObject = {"means": old_mean.tolist(), "cov": old_cov.tolist(), "mix": old_mix.tolist()}
                return modelObject, output

            responsibility = self.expectation(k, inputData, old_mean, old_cov, old_mix)
            mean, cov, mix = self.maximization(k, responsibility, inputData, old_mean)
            old_mean = mean
            old_cov = cov
            old_mix = mix
