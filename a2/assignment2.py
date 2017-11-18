from cv2 import calcHist, imread, imwrite
from em import EM
import json
from k_means import KMeans
import matplotlib.pyplot as plt
from sys import argv
from seeder import Seeder
from utils import process
import uuid


k = 2  # Segment foreground and background
sessionId = str(uuid.uuid4())


def output_dict_to_json(file_name, model):
    with open(file_name, "w") as fp:
        json.dump(model, fp)


def output_segmentation(file_name, images):
    masked_img, masked_img_inv, seg_img, seg_im_inv = images
    filte_title, extension = file_name.split(".")
    mask_file_title = "output/" + sessionId + "-" + filte_title + "-mask-" + "." + extension
    mask_inv_file_title = "output/" + sessionId + "-" + filte_title + "-mask-inv-" + "." + extension
    seg_file_title = "output/" + sessionId + "-" + filte_title + "-seg-" + "." + extension
    seg_inv_file_title = "output/" + sessionId + "-" + filte_title + "-seg-inv-" + "." + extension
    imwrite(mask_file_title, masked_img)
    imwrite(mask_inv_file_title, masked_img_inv)
    imwrite(seg_file_title, seg_img)
    imwrite(seg_inv_file_title, seg_im_inv)


def plot_histogram(image, channel_index, bin_count, hist_range):
    color = ('b', 'g', 'r')
    histr = calcHist([image], [channel_index], None, [bin_count], hist_range)
    plt.plot(histr, color=color[channel_index])
    plt.show()


def plot_histogram_full(image, bin_count, hist_range):
    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        histr = calcHist([image], [i], None, [bin_count], hist_range)
        plt.plot(histr, color=col)
        plt.xlim([0, 256])
    plt.show()


def main():
    file_name = argv[1]
    raw_input_data = imread(file_name)

    preprocessing_info = {
        "type": ["blur"],
        "kernel": [(5, 5)]
    }
    input_data = process(raw_input_data, preprocessing_info)
    color_quantization = 4

    mean_seeder = Seeder(k)
    seed_mean, quantized_image = mean_seeder.ilea_whelan_quantization(input_data, k + 5, color_quantization)

    postprocessing_info = {
        "type": ["blur", "max", "median"],
        "kernel": [(5, 5), (2, 2), 5]
    }

    em = EM()
    em_model_object, image = em.run(k,
                                    input_data,
                                    seed_mean=seed_mean,
                                    postprocessing_info=postprocessing_info,
                                    verbose=True)

    # km = KMeans()
    # em_model_object, image = km.run(k,
    #                                 input_data,
    #                                 seed_mean=seed_mean,
    #                                 postprocessing_info=postprocessing_info,
    #                                 verbose=True)

    output_dict_to_json("model.json", em_model_object)
    output_segmentation(file_name, image)


if __name__ == "__main__":
    main()
