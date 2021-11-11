import pandas as pd
import numpy as np

import constants as const

class Data():
    def __init__(self):
        """
        Loads the dataset saved as const.TRAIN_DATASET with pandas and separates
        the lead role column from the rest of the data, as well as making lead
        role boolean where female=1, male=0.

        self.x_train and self.y_train are np.arrays of type float32
        """

        csv = pd.read_csv(const.TRAIN_DATASET)

        # Convert the rest of the data into a numpy array and strip lead role (last column) from array
        self.x_train = np.array(csv)[:, 0:-1]

        # Convert 'Lead' attribute to string, then to boolean and set as y_train
        csv["Lead"] = csv["Lead"].astype("string")  # dtype becomes 'string' from 'object'
        self.y_train = csv['Lead'].map({'Female': 1, 'Male': 0})  # Convert strings to bools

        # Convert data into float32 so keras doesn't have a seizure using it.
        self.x_train = np.float32(self.x_train)
        self.y_train = np.float32(self.y_train)


if __name__ == "__main__":
    x = Data()
