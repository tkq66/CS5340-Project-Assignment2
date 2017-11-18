from cv2 import calcHist
from k_means import KMeans
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
        print(((h * w) / self.__k))

        im_color = ("r", "g", "b")
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
            im_channel = im_color.index(color)
            peak_pixels_index = np.where(image[:, :, im_channel] == intensity)
            peak_pixels = image[peak_pixels_index]
            seed_mean[i] = np.average(peak_pixels, axis=0)
        return seed_mean

    def ilea_whelan_quantization(self, image, R, quantization_level, seed_mean=None, patience=5, delta=1e-6, verbose=False):
        km = KMeans()
        _, quantized_image = km.run(quantization_level, image, quantize=True, seed_mean=seed_mean, patience=patience, delta=delta, verbose=True)
        raise ValueError("FDS")
        seed_mean = self.ilea_whelan_heuristic(quantized_image, R)
        return seed_mean

    def __get_histogram(self, image, channel_index):
        bin_count = 256
        hist_range = [0, 256]
        hist = calcHist([image], [channel_index], None, [bin_count], hist_range)
        return hist.ravel()