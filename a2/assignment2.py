from cv2 import imread
import numpy as np
from sys import argv

k = 2


def initialize():
    pass


def expectation():
    pass


def maximization():
    pass


def main():
    fileName = argv[1]
    inputData = np.rollaxis(imread(fileName), 2, 0)
    channels, height, width = inputData.shape


if __name__ == "__main__":
    main()
