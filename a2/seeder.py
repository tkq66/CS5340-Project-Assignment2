from cv2 import calcHist, kmeans, TERM_CRITERIA_EPS, TERM_CRITERIA_MAX_ITER, KMEANS_RANDOM_CENTERS
from k_means import KMeans
from math import sqrt
import numpy as np


class Seeder:

    def __init__(self, k):
        self.__k = k

    def forgy(self, image):
        height, width, channel = image.shape
        h = np.random.choice(np.arange(height), self.__k, replace=False)
        w = np.random.choice(np.arange(width), self.__k, replace=False)
        seed_mean = image[h, w, :]
        return seed_mean

    def naive_k_means(self, image, seed_mean=None, patience=5, delta=1e-6, verbose=False):
        km = KMeans()
        k_means_model_object, image = km.run(self.__k, image, seed_mean=seed_mean, patience=patience, delta=delta, verbose=False)
        seed_mean = k_means_model_object["means"]
        return seed_mean

    def ilea_whelan_heuristic(self, image, R):
        h, w, c = image.shape
        assert R > self.__k
        assert R < ((h * w) / self.__k)

        im_color = ("b", "g", "r")
        hist_color = ("b", "g", "r")
        peaks = []
        for i, color in enumerate(hist_color):
            hist = self.__get_histogram(image, i)
            hist_ref = [(intensity, hist[intensity]) for intensity in range(hist.size)]
            step_size = int(np.ceil(hist.size / R))
            r_split_hist = [hist_ref[i:i + step_size] for i in range(0, hist.size, step_size)]
            channel_peaks = [max(split, key=lambda x: x[1]) for split in r_split_hist]
            peaks += [(intensity, frequency, color) for intensity, frequency in channel_peaks]

        seed_mean = np.empty((self.__k, len(im_color)))
        ranked_peaks = sorted(peaks, key=lambda x: x[1], reverse=True)
        for i in range(self.__k):
            intensity, frequency, color = ranked_peaks[i]
            im_channel = hist_color.index(color)
            peak_pixels_index = np.where(image[:, :, im_channel] == intensity)
            peak_pixels = image[peak_pixels_index]
            seed_mean[i] = np.average(peak_pixels, axis=0)
        return seed_mean

    def ilea_whelan_quantization(self, image, R, quantization_level, seed_mean=None, patience=5, delta=1e-6, verbose=False):
        quantized_image = self.__get_color_quantized_image(image, quantization_level)
        seed_mean = self.ilea_whelan_heuristic(quantized_image, R)
        return seed_mean, quantized_image

    def bardlley_fayaad(self, image, J, patience=5, delta=1e-6, verbose=False):
        h, w, c = image.shape
        assert J > 0
        assert J < (h * w)

        h, w, c = image.shape
        image_data = image.reshape(h * w, c)
        data_split = np.array_split(image_data, J)
        candidate_mean_list = np.empty((J, self.__k, c))
        for j in range(J):
            j_data = data_split[j]
            candidate_mean = self.naive_k_means(j_data)
            candidate_mean_list[j] = candidate_mean
        mean_data = candidate_mean_list.copy()
        final_mean_list = np.empty((J, self.__k, c))
        for j in range(J):
            final_mean = self.naive_k_means(mean_data, seed_mean=candidate_mean_list[j])
            final_mean_list[j] = final_mean
        km = KMeans()
        distortion_list = np.empty(J)
        for j in range(J):
            distortion_list[j] = km.get_distortion(self.__k, final_mean_list[j], mean_data)
        seed_mean_index = np.argmin(distortion_list)
        seed_mean = final_mean_list[seed_mean_index]
        return seed_mean

    def __get_histogram(self, image, channel_index):
        bin_count = 256
        hist_range = [0, 256]
        hist = calcHist([image], [channel_index], None, [bin_count], hist_range)
        return hist.ravel()

    def __get_color_quantized_image(self, image, quantization_level):
        transformed_image = np.float32(image)
        # define criteria, number of clusters(K) and apply kmeans()
        criteria = (TERM_CRITERIA_EPS + TERM_CRITERIA_MAX_ITER, 10, 1.0)
        ret, label, center = kmeans(transformed_image, quantization_level, None, criteria, 10, KMEANS_RANDOM_CENTERS)

        # Now convert back into uint8, and make original image
        center = np.uint8(center)
        res = center[label.flatten()]
        quant = res.reshape((image.shape))
        return quant
