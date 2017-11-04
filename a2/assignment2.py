from cv2 import imread
import numpy as np
from sys import argv


def initialize(k, inputData):
    height, width, channels = inputData.shape
    initMixCoeff = None

    h = np.random.choice(np.arange(height), k, replace=False)
    w = np.random.choice(np.arange(width), k, replace=False)
    initMean = inputData[h, w, :]

    initCov = np.random.uniform(0, 3, size=(k, channels, channels))

    for i,x in enumerate(initCov):
        x = x.T.dot(x)
        initCov[i]=x

    print(initCov)
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
