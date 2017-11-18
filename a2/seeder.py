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
        k_means_model_object, images = km.run(self.__k, image, seed_mean=seed_mean, patience=patience, delta=delta, verbose=False)
        seed_mean = k_means_model_object["means"]
        return seed_mean

    def ilea_whelan_heuristic(self, image, R):
        h, w, c = image.shape
        assert R > self.__k
        assert R < ((h * w) / self.__k)

        im_color = ("r", "g", "b")
        hist_color = ("b", "g", "r")
        peaks = []
        for i, color in enumerate(hist_color):
            hist = self.__get_histogram(image, i)
            r_split_hist = np.array(np.array_split(hist, R))
            channel_peaks = [np.max(split) for split in r_split_hist]
            peaks += [(peak, color) for peak in channel_peaks]

        seed_mean = np.empty((self.__k, len(im_color)))
        ranked_peaks = sorted(peaks, key=lambda x: x[0], reverse=True)
        for i in range(self.__k):
            peak, color = ranked_peaks[i]
            im_channel = im_color.index(color)
            peak_pixels_index = np.where(image[:, :, im_channel] == peak)
            peak_pixels = image[peak_pixels_index]
            seed_mean[i] = np.average(peak_pixels, axis=0)
        return seed_mean

    def ilea_whelan_quantization(self, image, R):
        pass

    def __get_histogram(self, image, channel_index):
        bin_count = 256
        hist_range = [0, 256]
        hist = calcHist([image], [channel_index], None, [bin_count], hist_range)
        return hist.ravel()
