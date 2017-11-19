import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
from utils import process


class EM:

    def __init__(self):
        pass

    def initializeMean(self, k, n, input_data):
        # Initialize the mean
        i = np.random.choice(np.arange(n), k, replace=False)
        return input_data[i, :]

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

    def initialize(self, k, input_data):
        n, channels = input_data.shape
        initMean = self.initializeMean(k, n, input_data)
        initCov = self.initializeCov(k, channels)
        initMixCoeff = self.initializeMix(k)
        return initMean, initCov, initMixCoeff

    def expectation(self, k, input_data, mean, cov, mixCoeff):
        n, c = input_data.shape
        responsibility = None
        numerator = np.empty((k, n))
        for i in range(k):
            numerator[i] = mixCoeff[i] * multivariate_normal.pdf(input_data, mean=mean[i], cov=cov[i])
        denominator = numerator.sum(axis=0)  # for each N add values of K
        responsibility = numerator / denominator  # denominator.T is a column vector thus can numerator divide by column
        return responsibility

    def maximization(self, k, responsibility, input_data, means):
        n, c = input_data.shape
        # for each K add values of N
        N_k = responsibility.sum(axis=1)
        # CALCULATING NEW MEANS
        tempMean = np.empty((k, n, c))
        for i in range(k):
            tempMean[i] = (input_data.T * responsibility[i]).T
        new_means = (tempMean.sum(axis=1).T / N_k).T

        # CALCULATING NEW COVARIANCES
        new_cov = np.empty((k, c, c))
        for i in range(k):
            difference = input_data - new_means[i]
            mod = (responsibility[i] * difference.T) / N_k[i]
            new_cov[i] = mod.dot(difference)

        # CALCULATING NEW MIX COEF
        new_mixCoeff = N_k / n

        return new_means, new_cov, new_mixCoeff

    def evaluateLogLikelihood(self, k, input_data, mean, cov, mixCoeff):
        n, c = input_data.shape
        auxInput = np.zeros(n)
        for i in range(k):
            auxInput += mixCoeff[i] * multivariate_normal.pdf(input_data, mean=mean[i], cov=cov[i])
        logLikelihood = np.sum(np.log(auxInput))

        return -logLikelihood

    def segmentImage(self, k, input_data, img_shape, mean, cov, postprocessing_info=None):
        if img_shape is None:
            return
        n, c = input_data.shape
        image = input_data.reshape(img_shape)
        # For each pixel, calculate probability for each k
        rawKImageClassProb = np.empty((k, n))
        for i in range(k):
            rawKImageClassProb[i] = multivariate_normal.pdf(input_data, mean=mean[i], cov=cov[i])
        # Choose the k given the max probability
        pixelsClassAssignment = np.argmax(rawKImageClassProb, axis=0)
        # Black and white mask
        mask = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        # Apply the mask of k onto the pixel
        rawMaskedImage = mask[pixelsClassAssignment].reshape(img_shape)
        maskedImage = process(rawMaskedImage, postprocessing_info) if postprocessing_info is not None else rawMaskedImage
        maskedImageInverted = 1 - maskedImage.astype(np.uint8)
        segmentedImage = np.multiply(image, maskedImage)
        segmentedImageInverted = np.multiply(image, maskedImageInverted)
        return maskedImage * 255, maskedImageInverted * 255, segmentedImage, segmentedImageInverted

    def run(self, k, image, seed_mean=None, postprocessing_info=None, patience=5, delta=1e-6, verbose=False):
        assert k > 0
        assert isinstance(image, np.ndarray)
        assert (len(image.shape) == 3) or (len(image.shape) == 2)

        input_data = None
        img_shape = None
        if len(image.shape) == 3:
            height, width, channels = image.shape
            img_shape = image.shape
            input_data = image.reshape(height * width, channels)
        else:
            input_data = image

        oldLogLikelihood = 1000000
        logLikelihood = 1000000
        patienceCounter = 0

        # Train data, assuming convergen4ce criteria is logLikelihood = 0
        n, c = input_data.shape
        _, old_cov, old_mix = self.initialize(k, input_data)
        old_mean = self.initializeMean(k, n, input_data) if seed_mean is None else np.asarray(seed_mean)
        while True:
            try:
                logLikelihood = self.evaluateLogLikelihood(k, input_data, old_mean, old_cov, old_mix)
            except np.linalg.LinAlgError:
                if verbose:
                    print("New covariance matrix is singular. Resetting...")
                old_cov = self.initializeCov(k, channels)
                continue

            output = self.segmentImage(k, input_data, img_shape, old_mean, old_cov, postprocessing_info=postprocessing_info)

            # Check for convergence and output the model and the file
            diff = abs(oldLogLikelihood - logLikelihood)
            if verbose and img_shape is not None:
                plt.imshow(output[0] / 255)
                plt.pause(0.1)
                plt.draw()
            if verbose:
                print(diff)
            if diff < delta:
                patienceCounter += 1
            else:
                patienceCounter = 0
                oldLogLikelihood = logLikelihood
            if patienceCounter > patience:
                modelObject = {"means": old_mean.tolist(), "cov": old_cov.tolist(), "mix": old_mix.tolist()}
                return modelObject, output

            responsibility = self.expectation(k, input_data, old_mean, old_cov, old_mix)
            mean, cov, mix = self.maximization(k, responsibility, input_data, old_mean)
            old_mean = mean
            old_cov = cov
            old_mix = mix
