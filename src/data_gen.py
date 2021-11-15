import pandas as pd
import numpy as np
import os

import src.constants as const

class Data():
    def __init__(self, train_dataset, test_dataset):
        """
        Loads the dataset saved as const.TRAIN_DATASET with pandas and separates
        the lead role column from the rest of the data, as well as making lead
        role boolean where female=1, male=0.

        self.x_train and self.y_train are np.arrays of type float32
        """

        if not os.path.exists(train_dataset) or not os.path.exists(train_dataset):
            print("The file given does not exist, please check the path")
            raise FileNotFoundError

        train_data = pd.read_csv(train_dataset)
        test_data = pd.read_csv(test_dataset)

        # Convert the rest of the data into a numpy array and strip lead role (last column) from array
        self.x_train = np.array(train_data)[:, 0:-1]

        # Convert 'Lead' attribute to string, then to boolean and set as y_train
        train_data["Lead"] = train_data["Lead"].astype("string")  # dtype becomes 'string' from 'object'
        self.y_train = train_data['Lead'].map({'Female': 1, 'Male': 0})  # Convert strings to bools
        print(np.sum(self.y_train))

        # Convert data into float32 so keras doesn't have a seizure using it.
        self.x_train = np.float32(self.x_train)
        self.y_train = np.float32(self.y_train)

        self.normalized_x_train = self.x_train / np.amax(self.x_train)


        self.x_train = self.x_train[0:800]
        self.y_train = self.y_train[0:800]
        self.x_test = self.x_train[801:]
        self.y_test = self.y_train[801:]
        # self.x_test = np.array(test_data)

if __name__ == "__main__":
    x = Data(const.TRAIN_DATASET, const.TEST_DATASET)
