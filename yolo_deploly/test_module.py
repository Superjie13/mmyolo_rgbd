# This script is used to test torch.onnx.export function:
# 1. Given a dict of input tensor, export the model to onnx format (Failed)  # Note: torch.jit.trace does not support dict input
# 2. Given a list of input tensor, export the model to onnx format


import os
from typing import Dict, List
import torch
import torch.nn as nn


# Create a simple model with a single convolutional layer
class TestModel_Dict(nn.Module):
    def __init__(self):
        super(TestModel_Dict, self).__init__()
        self.conv_x = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3, stride=1, padding=1)
        self.conv_y = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3, stride=1, padding=1)
        self.ReLU = nn.ReLU()

    def forward(self, input: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = self.conv_x(input['x'])
        x = self.ReLU(x)
        y = self.conv_y(input['y'])
        y = self.ReLU(y)
        return x + y


class TestModel_List(nn.Module):
    def __init__(self):
        super(TestModel_List, self).__init__()
        self.conv_x = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3, stride=1, padding=1)
        self.conv_y = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3, stride=1, padding=1)
        self.ReLU = nn.ReLU()

    def forward(self, input: List[torch.Tensor]) -> torch.Tensor:
        x = self.conv_x(input[0])
        x = self.ReLU(x)
        y = self.conv_y(input[1])
        y = self.ReLU(y)
        return x + y


if __name__ == '__main__':
    # Create a model instance
    model_dict = TestModel_Dict()
    # Set the model to evaluation
    model_dict.eval()

    # Create a dict of input tensor
    input_dict = {'x': torch.zeros(1, 3, 224, 224), 'y': torch.ones(1, 3, 224, 224)}
    output = model_dict(input_dict)
    print(output)

    # Export the model to onnx format
    # torch.onnx.export(model_dict, input_dict, 'test_model_dict.onnx', verbose=True)


    # Create a model instance
    model_list = TestModel_List()
    # Set the model to evaluation
    model_list.eval()

    # Create a list of input tensor
    input_list = [torch.zeros(1, 3, 224, 224), torch.ones(1, 3, 224, 224)]
    # Export the model to onnx format
    torch.onnx.export(model_list, input_list, 'test_model_list.onnx', verbose=True, input_names=['x',], output_names=['output'])
