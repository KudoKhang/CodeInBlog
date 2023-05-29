import numpy as np
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential

ALPHA_DICT = {
    0: "A",
    1: "B",
    2: "C",
    3: "D",
    4: "E",
    5: "F",
    6: "G",
    7: "H",
    8: "K",
    9: "L",
    10: "M",
    11: "N",
    12: "P",
    13: "R",
    14: "S",
    15: "T",
    16: "U",
    17: "V",
    18: "X",
    19: "Y",
    20: "Z",
    21: "0",
    22: "1",
    23: "2",
    24: "3",
    25: "4",
    26: "5",
    27: "6",
    28: "7",
    29: "8",
    30: "9",
    31: "Background",
}


class CNN_Model(object):
    def __init__(self):
        # Building model
        self._build_model()

        self.model.compile(
            loss="categorical_crossentropy",
            optimizer=optimizers.legacy.Adam(1e-3),
            metrics=["acc"],
        )

    def _build_model(self):
        # CNN model
        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), padding="same", activation="relu", input_shape=(28, 28, 1)))
        self.model.add(Conv2D(32, (3, 3), activation="relu"))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
        self.model.add(Conv2D(64, (3, 3), activation="relu"))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
        self.model.add(Conv2D(64, (3, 3), activation="relu"))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Flatten())
        self.model.add(Dense(512, activation="relu"))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(32, activation="softmax"))
