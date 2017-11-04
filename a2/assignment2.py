from cv2 import imread
import numpy as np
from sys import argv


def initialize():
    pass


def expectation():
    pass


def maximization():
    pass


def main():
    k = argv[1]
    fileName = argv[2]
    inputData = np.rollaxis(imread(fileName), 2, 0)
    channels, height, width = inputData.shape
    h = np.random.randint(low=0,high=height)
    w = np.random.randint(low=0, high=height)
    pix = inputData[h,w,:]

if __name__ == "__main__":
    main()
