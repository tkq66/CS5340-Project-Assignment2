from cv2 import imread, imwrite
from em import EM
import json
from sys import argv
import uuid

k = 2  # Segment foreground and background
sessionId = str(uuid.uuid4())


def outputDicttoJson(fileName, model):
    with open(fileName, "w") as fp:
        json.dump(model, fp)


def outputSegmentation(fileName, images):
    maskedImage, maskedImageInverted, segmentedImage, segmentedImageInverted = images
    filteTitle, extension = fileName.split(".")
    maskFileTitle = "output/" + sessionId + "-" + filteTitle + "-mask-" + "." + extension
    maskInvFileTitle = "output/" + sessionId + "-" + filteTitle + "-mask-inv-" + "." + extension
    segFileTitle = "output/" + sessionId + "-" + filteTitle + "-seg-" + "." + extension
    segInvFileTitle = "output/" + sessionId + "-" + filteTitle + "-seg-inv-" + "." + extension
    imwrite(maskFileTitle, maskedImage)
    imwrite(maskInvFileTitle, maskedImageInverted)
    imwrite(segFileTitle, segmentedImage)
    imwrite(segInvFileTitle, segmentedImageInverted)


def main():
    fileName = argv[1]
    inputData = imread(fileName)

    em = EM()
    modelObject, images = em.run(k, inputData, verbose=True)

    outputDicttoJson("model.json", modelObject)
    outputSegmentation(fileName, images)


if __name__ == "__main__":
    main()
