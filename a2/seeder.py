from cv2 import calcHist, kmeans, TERM_CRITERIA_EPS, TERM_CRITERIA_MAX_ITER, KMEANS_RANDOM_CENTERS
from k_means import KMeans
import numpy as np


class Seeder:

    FORGY_TYPE = "fg"
    NAIVE_KMEANS_TYPE = "nkm"
    BRADLEY_FAYAAD_TYPE = "bf"
    ILEA_WHELAN_HEURISTIC_TYPE = "iwh"
    ILEA_WHELAN_QUANTIZATION_TYPE = "iwq"

    def __init__(self, k):
        self.__k = k

    def forgy(self, image):
        height, width, channel = image.shape
        h = np.random.choice(np.arange(height), self.__k, replace=False)
        w = np.random.choice(np.arange(width), self.__k, replace=False)
        seed_mean = image[h, w, :]
        return seed_mean

    def naive_k_means(self,
                      image,
                      seed_mean=None,
                      patience=5,
                      delta=1e-6,
                      verbose=False):
        km = KMeans()
        k_means_model_object, image = km.run(self.__k,
                                             image,
                                             seed_mean=seed_mean,
                                             patience=patience,
                                             delta=delta,
                                             verbose=False)
        seed_mean = k_means_model_object["means"]
        return seed_mean

    def ilea_whelan_heuristic(self,
                              image,
                              R,
                              incl_spatial_relations=False,
                              spatial_selection_mode="avg"):
        h, w, c = image.shape
        assert R > self.__k
        assert R < ((h * w) / self.__k)

        hist_color = ("b", "g", "r")
        peaks = []
        for i, color in enumerate(hist_color):
            hist = self.__get_histogram(image, i)
            hist_ref = [(intensity, hist[intensity]) for intensity in range(hist.size)]
            step_size = int(np.ceil(hist.size / R))
            r_split_hist = [hist_ref[i:i + step_size] for i in range(0, hist.size, step_size)]
            channel_peaks = [max(split, key=lambda x: x[1]) for split in r_split_hist]
            peaks += [(intensity, frequency, color) for intensity, frequency in channel_peaks]

        seed_mean = None
        if incl_spatial_relations:
            seed_mean = np.empty((self.__k, c + 2))
        else:
            seed_mean = np.empty((self.__k, c))
        ranked_peaks = sorted(peaks, key=lambda x: x[1], reverse=True)
        for i in range(self.__k):
            intensity, frequency, color = ranked_peaks[i]
            im_channel = hist_color.index(color)
            peak_pixels_index = np.where(image[:, :, im_channel] == intensity)
            peak_pixels = image[peak_pixels_index]
            seed_colors = np.average(peak_pixels, axis=0).flatten()
            if incl_spatial_relations:
                seed_pixel_loc = None
                if spatial_selection_mode == "avg":
                    seed_pixel_loc = np.average(np.matrix(peak_pixels_index), axis=1).flatten().astype(np.uint8)
                elif spatial_selection_mode == "rand":
                    random_id = np.random.randint(10, size=1)
                    seed_pixel_loc = np.matrix(peak_pixels_index)[:, random_id].flatten()
                else:
                    seed_pixel_loc = np.average(np.matrix(peak_pixels_index), axis=1).flatten().astype(np.uint8)
                loc_len = seed_pixel_loc.size
                color_len = seed_colors.size
                seed_mean[i, 0:loc_len] = seed_pixel_loc
                seed_mean[i, loc_len:loc_len + color_len] = seed_colors
            else:
                seed_mean[i] = seed_colors
        return seed_mean

    def ilea_whelan_quantization(self,
                                 image,
                                 R,
                                 quantization_level,
                                 incl_spatial_relations=False,
                                 spatial_selection_mode="avg",
                                 seed_mean=None,
                                 verbose=False):
        quantized_image = self.__get_color_quantized_image(image, quantization_level)
        seed_mean = self.ilea_whelan_heuristic(quantized_image,
                                               R,
                                               incl_spatial_relations=incl_spatial_relations,
                                               spatial_selection_mode=spatial_selection_mode)
        return seed_mean, quantized_image

    def bardlley_fayaad(self,
                        image,
                        J,
                        patience=5,
                        delta=1e-6,
                        verbose=False):
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

    def __get_histogram(self,
                        image,
                        channel_index):
        bin_count = 256
        hist_range = [0, 256]
        hist = calcHist([image], [channel_index], None, [bin_count], hist_range)
        return hist.ravel()

    def __get_color_quantized_image(self,
                                    image,
                                    quantization_level):
        transformed_image = np.float32(image)
        # define criteria, number of clusters(K) and apply kmeans()
        criteria = (TERM_CRITERIA_EPS + TERM_CRITERIA_MAX_ITER, 10, 1.0)
        ret, label, center = kmeans(transformed_image, quantization_level, None, criteria, 10, KMEANS_RANDOM_CENTERS)

        # Now convert back into uint8, and make original image
        center = np.uint8(center)
        res = center[label.flatten()]
        quant = res.reshape((image.shape))
        return quant

    def __get_peaks_in_array(self, input_array):
        are_peaks = np.r_[True, input_array[1:] > input_array[:-1]] & np.r_[input_array[:-1] > input_array[1:], True]
        peak_vals_index = np.where(are_peaks == True)[0]
        peak_reference = [(i, input_array[i]) for i in peak_vals_index]
        return peak_reference
