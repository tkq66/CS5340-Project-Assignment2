from cv2 import imread, imwrite
from em import EM
import json
from k_means import KMeans
from sys import argv
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


def main():
    file_name = argv[1]
    input_data = imread(file_name)

    km = KMeans()
    k_means_model_object, images = km.run(k, input_data)
    seed_mean = k_means_model_object["means"]
    em = EM()
    em_model_object, images = em.run(k, input_data, seed_mean=seed_mean, verbose=True)

    output_dict_to_json("model.json", em_model_object)
    output_segmentation(file_name, images)


if __name__ == "__main__":
    main()
