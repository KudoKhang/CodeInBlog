import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, ".")
from models.tf_model import ALPHA_DICT, CNN_Model

input_data = cv2.imread("data/char_G.png", 0)
input_data = np.expand_dims(input_data, 0)
print(input_data.shape)

model_recognize_character = CNN_Model().model
model_recognize_character.load_weights("checkpoints/classify_character.h5")

result = model_recognize_character.predict_on_batch(input_data)
idx = np.argmax(result)

print(result)
YELLOW = "\033[1;33m"
print(YELLOW, "Result: ---", ALPHA_DICT[idx], "---")
