import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
from utils import process, conv_3d_to_5d, conv_5d_to_3d


class EM:

    def __init__(self):
        pass

    def initialize_mean(self, k, n, input_data):
        # Initialize the mean
        i = np.random.choice(np.arange(n), k, replace=False)
        return input_data[i, :]

    def initialize_cov(self, k, channels, spread=50):
        # Initialzie covariance
        init_cov = np.random.uniform(0, spread, size=(k, channels, channels))
        for i, x in enumerate(init_cov):
            x = x.T.dot(x)
            init_cov[i] = x
        return init_cov

    def initialize_mix(self, k):
        # Initialize mixing coefficient
        return np.repeat((1 / k), k)

    def initialize(self, k, input_data):
        n, c = input_data.shape
        init_mean = self.initialize_mean(k, n, input_data)
        init_cov = self.initialize_cov(k, c)
        init_mix_coeff = self.initialize_mix(k)
        return init_mean, init_cov, init_mix_coeff

    def expectation(self, k, input_data, mean, cov, mix_coeff):
        n, c = input_data.shape
        responsibility = None
        numerator = np.empty((k, n))
        for i in range(k):
            numerator[i] = mix_coeff[i] * multivariate_normal.pdf(input_data, mean=mean[i], cov=cov[i])
        denominator = numerator.sum(axis=0)  # for each N add values of K
        responsibility = numerator / denominator  # denominator.T is a column vector thus can numerator divide by column
        return responsibility

    def maximization(self, k, responsibility, input_data, means):
        n, c = input_data.shape
        # for each K add values of N
        N_k = responsibility.sum(axis=1)
        # CALCULATING NEW MEANS
        temp_mean = np.empty((k, n, c))
        for i in range(k):
            temp_mean[i] = (input_data.T * responsibility[i]).T
        new_means = (temp_mean.sum(axis=1).T / N_k).T

        # CALCULATING NEW COVARIANCES
        new_cov = np.empty((k, c, c))
        for i in range(k):
            difference = input_data - new_means[i]
            mod = (responsibility[i] * difference.T) / N_k[i]
            new_cov[i] = mod.dot(difference)

        # CALCULATING NEW MIX COEF
        new_mix_coeff = N_k / n

        return new_means, new_cov, new_mix_coeff

    def evaluate_log_likelihood(self, k, input_data, mean, cov, mix_coeff):
        n, c = input_data.shape
        aux_input = np.zeros(n)
        for i in range(k):
            aux_input += mix_coeff[i] * multivariate_normal.pdf(input_data, mean=mean[i], cov=cov[i])
        log_likelihood = np.sum(np.log(aux_input))

        return -log_likelihood

    def segment_image(self, k, input_data, img_shape, mean, cov, incl_spatial_relations=False, postprocessing_info=None):
        if img_shape is None:
            return
        n, c = input_data.shape
        image = None
        if incl_spatial_relations:
            image = conv_5d_to_3d(input_data, img_shape)
        else:
            image = input_data.reshape(img_shape)
        # For each pixel, calculate probability for each k
        raw_k_image_class_prob = np.empty((k, n))
        for i in range(k):
            raw_k_image_class_prob[i] = multivariate_normal.pdf(input_data, mean=mean[i], cov=cov[i])
        # Choose the k given the max probability
        pixels_class_assignment = np.argmax(raw_k_image_class_prob, axis=0)
        # Black and white mask
        mask = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        # Apply the mask of k onto the pixel
        raw_masked_image = mask[pixels_class_assignment].reshape(img_shape)
        masked_image = process(raw_masked_image, postprocessing_info) if postprocessing_info is not None else raw_masked_image
        masked_image_inverted = 1 - masked_image.astype(np.uint8)
        segmented_image = np.multiply(image, masked_image)
        segmented_image_inverted = np.multiply(image, masked_image_inverted)
        return masked_image * 255, masked_image_inverted * 255, segmented_image, segmented_image_inverted

    def run(self, k, image, incl_spatial_relations=False, seed_mean=None, postprocessing_info=None, patience=5, delta=1e-6, verbose=False):
        assert k > 0
        assert isinstance(image, np.ndarray)
        assert (len(image.shape) == 3) or (len(image.shape) == 2)

        input_data = None
        img_shape = None
        if len(image.shape) == 3:
            height, width, channels = image.shape
            img_shape = image.shape
            if incl_spatial_relations:
                input_data = conv_3d_to_5d(image)
            else:
                input_data = image.reshape(height * width, channels)
        else:
            input_data = image

        old_log_likelihood = 1000000
        log_likelihood = 1000000
        patience_counter = 0

        # Train data, assuming convergen4ce criteria is log_likelihood = 0
        n, c = input_data.shape
        old_cov = self.initialize_cov(k, c)
        old_mix = self.initialize_mix(k)
        old_mean = self.initialize_mean(k, n, input_data) if seed_mean is None else np.asarray(seed_mean)
        while True:
            try:
                log_likelihood = self.evaluate_log_likelihood(k, input_data, old_mean, old_cov, old_mix)
            except np.linalg.LinAlgError:
                if verbose:
                    print("New covariance matrix is singular. Resetting...")
                old_cov = self.initialize_cov(k, channels)
                continue

            output = self.segment_image(k, input_data, img_shape, old_mean, old_cov, incl_spatial_relations=incl_spatial_relations, postprocessing_info=postprocessing_info)

            # Check for convergence and output the model and the file
            diff = abs(old_log_likelihood - log_likelihood)
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
                old_log_likelihood = log_likelihood
            if patience_counter > patience:
                model_object = {"means": old_mean.tolist(), "cov": old_cov.tolist(), "mix": old_mix.tolist()}
                return model_object, output

            responsibility = self.expectation(k, input_data, old_mean, old_cov, old_mix)
            mean, cov, mix = self.maximization(k, responsibility, input_data, old_mean)
            old_mean = mean
            old_cov = cov
            old_mix = mix
