import matplotlib.pyplot as plt
import numpy as np
from utils import process


class KMeans:

    def __init__(self):
        pass

    def initialize_mean(self, k, n, input_data):
        # Initialize the mean
        i = np.random.choice(np.arange(n), k, replace=False)
        return input_data[i, :]

    def assignment(self, k, mean, input_data):
        n, c = input_data.shape
        data = input_data.copy()
        # Choose the enarest distance
        distance = self.calculate_distance(k, mean, data)
        max_index = (np.arange(n) * k) + np.argmax(distance, axis=0)
        # Set all values to zero and assign 1 to the indexes informed by argmax
        r = np.full(distance.shape, 0)
        flatten_r = r.ravel()
        flatten_r[max_index] = 1
        r = flatten_r.reshape(distance.T.shape)
        return r

    def update(self, k, r, input_data, old_mean):
        data = input_data.copy()
        new_mean = np.empty(old_mean.shape)
        for i in range(k):
            assignment_sum = np.sum(r[:, i])
            if assignment_sum == 0:
                new_mean[i] = old_mean[i]
                continue
            masking = r[:, i] * data.T
            new_mean[i] = np.sum(masking, axis=1) / np.sum(r[:, i])
        return new_mean

    def calculate_distortion(self, k, r, mean, input_data):
        data = input_data.copy()
        distance = self.calculate_distance(k, mean, data)
        distortion = np.sum(np.multiply(distance, r.T))
        return abs(distortion)

    def calculate_distance(self, k, mean, data):
        total_pixels, c = data.shape
        distance = np.empty((k, total_pixels))
        for i in range(k):
            d = (data - mean[i]) ** 2
            distance[i] = np.sum(d, axis=1)
        return distance

    def get_distortion(self, k, mean, image):
        input_data = None
        if len(image.shape) == 3:
            h, w, c = image.shape
            input_data = image.reshape(h * w, c)
        else:
            input_data = image
        r = self.assignment(k, mean, input_data)
        distortion = self.calculate_distortion(k, r, mean, input_data)
        return distortion

    def segment_image(self, k, r, input_data, img_shape, postprocessing_info=None):
        if img_shape is None:
            return
        image = input_data.reshape(img_shape)
        mask = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        raw_masked_image = mask[r.T[0]].reshape(image.shape)
        masked_image = process(raw_masked_image, postprocessing_info) if postprocessing_info is not None else raw_masked_image
        masked_image_inverted = 1 - masked_image.astype(np.uint8)
        segmented_image = np.multiply(image, masked_image)
        segmented_image_inverted = np.multiply(image, masked_image_inverted)
        return masked_image * 255, masked_image_inverted * 255, segmented_image, segmented_image_inverted

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

        old_distortion = 1000000
        distortion = 1000000
        patience_counter = 0

        # Train data, assuming convergen4ce criteria is logLikelihood = 0
        n, c = input_data.shape
        old_mean = self.initialize_mean(k, n, input_data) if seed_mean is None else np.asarray(seed_mean)
        while True:
            r = self.assignment(k, old_mean, input_data)
            output = self.segment_image(k, r, input_data, img_shape, postprocessing_info=postprocessing_info)
            distortion = self.calculate_distortion(k, r, old_mean, input_data)

            # Check for convergence and output the model and the file
            diff = abs(distortion - old_distortion)
            if verbose and img_shape is not None:
                plt.imshow(output[0] / 255)
                plt.pause(0.1)
                plt.draw()
            if verbose:
                print(diff)
            if diff < delta:
                patience_counter += 1
            else:
                patience_counter = 0
                old_distortion = distortion
            if patience_counter > patience:
                model_object = {"means": old_mean.tolist()}
                return model_object, output

            mean = self.update(k, r, input_data, old_mean)
            old_mean = mean
