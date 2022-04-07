import logging
import numpy as np

error = 1e-5


def compare(input1, input2):
    input1 = np.ndarray.flatten(input1)
    input2 = np.ndarray.flatten(input2)
    if input1.size != input2.size:
        logging.error("Two inputs have different size!!!")
        exit(1)

    diff_num = 0

    for i in range(input1.size):
        diff = abs(input1[i] - input2[i])
        if diff > error:
            print("Index ", i, " diff: ", diff)
            diff_num += 1

    if diff_num == 0:
        print("All the elements are the same!!!")
