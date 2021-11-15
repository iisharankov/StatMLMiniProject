import os

# Find absolute path to this file so we can import data from anywhere in the repo
abs_path = os.path.dirname(os.path.abspath(__file__))

TRAIN_DATASET = abs_path + "/data/train.csv"
TEST_DATASET = abs_path + "/data/test.csv"


# CNN
CNN_LR = 0.5
CNN_EPOCHS = 1000
CNN_DISPLAY_EPOCHS = 50
CNN_V = 2 # Verbosity
