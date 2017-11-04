from cv2 import imread
import numpy as np
from sys import argv


def initialize(k, channels, input):
    initMean = np.zeros((k, channels))
    initCov = None
    initMixCoeff = None

    for i in range(k):
        

    return initMean, initCov, initMixCoeff


def expectation():
    pass


def maximization():
    pass


def main():
    k = argv[1]
    fileName = argv[2]
    inputData = imread(fileName)
    height, width, channels = inputData.shape
    initMean, initCov, initMixCoeff = initialize(k, inputData)


if __name__ == "__main__":
    main()
