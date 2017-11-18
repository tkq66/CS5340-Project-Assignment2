from cv2 import blur, GaussianBlur, medianBlur
import numpy as np


def process(image, processing_info):
    h, w, c = image.shape
    processed_image = image
    for i in range(len(processing_info["type"])):
        processing_type = processing_info["type"][i]
        processing_kernel = processing_info["kernel"][i]
        if processing_type == "median":
            processed_image = medianBlur(image.astype(np.uint8), processing_kernel)
        elif processing_type == "blur":
            processed_image = blur(image, processing_kernel)
        elif processing_type == "max":
            kernel = processing_kernel
            processed_image = max_pool(image, kernel)
    return processed_image


def max_pool(bw_image, kernel):
    h, w, c = bw_image.shape
    hh, ww = kernel
    feature_map_height = int(1 + np.ceil((h - hh) / hh))
    feature_map_width = int(1 + np.ceil((w - ww) / ww))

    pooled_image = bw_image
    for i in range(feature_map_height):
        for j in range(feature_map_width):
            w_start = j * ww
            w_end = w_start + ww
            if w_end > w:
                w_start = w - ww
                w_end = w
            h_start = i * hh
            h_end = h_start + hh
            if h_end > h:
                h_start = h - hh
                h_end = h
            receptive_field = pooled_image[h_start:h_end, w_start:w_end]
            avg = np.average(receptive_field.ravel())
            rounded = np.ceil(avg) if avg >= 0.5 else np.floor(avg)
            new_color = np.full(c, rounded)
            pooled_image[h_start:h_end, w_start:w_end] = new_color
    return pooled_image
