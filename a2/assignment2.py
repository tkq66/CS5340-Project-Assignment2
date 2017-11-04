from cv2 import imread
import numpy as np
from sys import argv


def initialize(k, inputData):
    channels, height, width = inputData.shape
    initCov = None
    initMixCoeff = None

    h = np.random.randint(low=0, high=height)
    w = np.random.randint(low=0, high=width)
    initMean = inputData[h, w, :]

    return initMean, initCov, initMixCoeff


def expectation():
    pass


def maximization():
    pass


def main():
    k = argv[1]
    fileName = argv[2]

    inputData = imread(fileName)
    initMean, initCov, initMixCoeff = initialize(k, inputData)

    print(initMean)


if __name__ == "__main__":
    main()
