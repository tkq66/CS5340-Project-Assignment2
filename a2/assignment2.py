from cv2 import imread
import numpy as np
from sys import argv


def initialize(k, inputData):
    height, width, channels = inputData.shape
    initCov = None
    initMixCoeff = None

    h = np.random.choice(np.arange(height), k, replace=False)
    w = np.random.choice(np.arange(width), k, replace=False)
    initMean = inputData[h, w, :]

    return initMean, initCov, initMixCoeff


def expectation():
    pass


def maximization():
    pass


def main():
    k = int(argv[1])
    fileName = argv[2]

    inputData = imread(fileName)
    initMean, initCov, initMixCoeff = initialize(k, inputData)

if __name__ == "__main__":
    main()
