import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearClassifier(nn.Module):

    def __init__(self, args):
        super(LinearClassifier, self).__init__()
        self.device = args.device
        self.batch_size = args.batch_size
        self.input_dimension = args.input_dimension
        self.input_dimension = args.hidden_dimension
        self.output_dimension = args.output_dimension

        # Encoder
        self.linear_1 = nn.Linear(self.input_dimension, self.hidden_dimension)
        self.linear_2 = nn.Linear(self.hidden_dimension, self.output_dimension)

    def forward(self, x):
        x1 = self.linear_1(x)
        x2 = self.linear_2(x1)

        return x2
