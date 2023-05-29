import sys

sys.path.insert(0, ".")


import cv2
import numpy as np
import torch
import torch.nn as nn
from converter import CNN_Model_Pytorch
from keras.models import load_model
from models.tf_model import ALPHA_DICT

torch_model = CNN_Model_Pytorch()


def infer():
    # Load the PyTorch model's state dictionary from the checkpoint file
    checkpoint = torch.load("checkpoints/classify_character.pt")
    torch_model.load_state_dict(checkpoint)

    # Put the PyTorch model in evaluation mode
    torch_model.eval()

    # Load your input data and preprocess it as necessary
    # input_data = np.random.rand(1, 28, 28, 1)  # replace with your actual input data
    input_data = cv2.imread("data/char_G.png", 0)
    input_data = np.expand_dims(input_data, 0)
    input_data = np.expand_dims(input_data, -1)
    input_data_2 = input_data.copy()

    input_tensor = torch.from_numpy(input_data.transpose(0, 3, 1, 2)).float()
    print(input_tensor.shape)

    # Pass the preprocessed input data through the PyTorch model
    output_tensor = torch_model.forward(input_tensor)

    # Use the output of the PyTorch model to make predictions or perform other downstream tasks
    predictions = output_tensor.detach().numpy()
    idx = np.argmax(predictions)
    print(predictions)
    YELLOW = "\033[1;33m"
    print(YELLOW, "Result: ---", ALPHA_DICT[idx], "---")


if __name__ == "__main__":
    infer()
