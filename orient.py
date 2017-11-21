#!/usr/bin/env python

import numpy as np
import pandas as pd
import sys


def read_file(fname):
    data = {}
    with open(fname, "r") as file:
        for i, line in enumerate(file):
            vector = line.split()
            data[i] = {"photo_id": vector[0],
                       "correct_orientation": vector[1],
                       "feature": vector[2:]}
#                       "r": np.array(vector[2::3]).reshape(8,8),
#                       "g": np.array(vector[3::3]).reshape(8,8),
#                       "b": np.array(vector[4::3]).reshape(8,8)}
    return data


if __name__ == "__main__":
#    train_fname = "train-data.txt"
#    test_fname = "test-data.txt"

    task, fname, model_file, model = sys.argv[1:]

    data = read_file(fname)

