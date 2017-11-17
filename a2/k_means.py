import matplotlib.pyplot as plt
import numpy as np


class KMeans:

    def __init__(self):
        pass

    def initialize_mean(self, k, height, width, input_data):
        # Initialize the mean
        h = np.random.choice(np.arange(height), k, replace=False)
        w = np.random.choice(np.arange(width), k, replace=False)
        return input_data[h, w, :]

    def assignment(self, k, mean, input_data):
        h, w, c = input_data.shape
        total_pixels = h * w
        data = input_data.reshape(total_pixels, c)
        # Choose the enarest distance
        distance = self.calculate_distance(k, mean, data)
        max_index = (np.arange(total_pixels) * 2) + np.argmax(distance, axis=0)
        # Set all values to zero and assign 1 to the indexes informed by argmax
        r = np.full(distance.shape, 0)
        flatten_r = r.ravel()
        flatten_r[max_index] = 1
        r = flatten_r.reshape(distance.T.shape).T
        return r

    def update(self, k, r, input_data, old_mean):
        h, w, c = input_data.shape
        total_pixels = h * w
        data = input_data.reshape(total_pixels, c)
        new_mean = np.empty(old_mean.shape)
        for i in range(k):
            masking = r[i] * data.T
            new_mean[i] = np.sum(masking, axis=1) / np.sum(r[i])
        return new_mean

    def calculate_distortion(self, k, r, mean, input_data):
        h, w, c = input_data.shape
        total_pixels = h * w
        data = input_data.reshape(total_pixels, c)
        distance = self.calculate_distance(k, mean, data)
        distortion = np.sum(np.multiply(distance, r))
        return abs(distortion)

    def calculate_distance(self, k, mean, flat_data):
        total_pixels, c = flat_data.shape
        distance = np.empty((k, total_pixels))
        for i in range(k):
            d = (flat_data - mean[i]) ** 2
            distance[i] = np.sum(d, axis=1)
        return distance

    def segment_image(self, k, r, input_data):
        mask = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        maskedImage = mask[r[0]].reshape(input_data.shape)
        maskedImageInverted = mask[r[1]].reshape(input_data.shape)
        segmentedImage = np.multiply(input_data, maskedImage)
        segmentedImageInverted = np.multiply(input_data, maskedImageInverted)
        return maskedImage * 255, maskedImageInverted * 255, segmentedImage, segmentedImageInverted

    def run(self, k, input_data, patience=5, delta=1e-6, verbose=False):
        assert k > 0
        assert isinstance(input_data, np.ndarray)
        assert len(input_data.shape) == 3

        old_distortion = 1000000
        distortion = 1000000
        patience_counter = 0

        # Train data, assuming convergen4ce criteria is logLikelihood = 0
        height, width, channels = input_data.shape
        old_mean = self.initialize_mean(k, height, width, input_data)
        while distortion != 0:
            r = self.assignment(k, old_mean, input_data)
            output = self.segment_image(k, r, input_data)
            # Check for convergence and output the model and the file
            distortion = self.calculate_distortion(k, r, old_mean, input_data)
            diff = distortion - old_distortion
            if verbose:
                plt.imshow(output[0] / 255)
                plt.pause(0.1)
                plt.draw()
                print(diff)
            if diff < delta:
                patience_counter += 1
            else:
                patience_counter = 0
                old_distortion = distortion
            if patience_counter > patience:
                modelObject = {"means": old_mean.tolist()}
                return modelObject, output
            mean = self.update(k, r, input_data, old_mean)
            old_mean = mean
