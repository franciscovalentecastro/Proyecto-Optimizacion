import math
import torch.nn as nn
import torch.nn.functional as F


class LinearClassifier(nn.Module):

    def __init__(self, args):
        super(LinearClassifier, self).__init__()
        self.device = args.device
        self.batch_size = args.batch_size
        self.input_dimension = args.input_dimension
        self.hidden_dimension = args.hidden_dimension
        self.output_dimension = args.output_dimension

        # Classifier
        self.linear_1 = nn.Linear(self.input_dimension, self.hidden_dimension)
        self.linear_2 = nn.Linear(self.hidden_dimension, self.output_dimension)

    def forward(self, x):
        # Pass through layers
        x1 = F.relu(self.linear_1(x))
        x2 = self.linear_2(x1)

        return x2


class ConvolutionalClassifier(nn.Module):

    def __init__(self, args):
        super(ConvolutionalClassifier, self).__init__()
        self.device = args.device
        self.batch_size = args.batch_size
        self.input_dimension = args.input_dimension
        self.hidden_dimension = args.hidden_dimension
        self.output_dimension = args.output_dimension

        # Calculate parameters
        self.image_dimension = int(math.sqrt(self.input_dimension))

        # Classifier
        self.conv_1 = nn.Conv2d(1, 8, kernel_size=5)
        self.conv_2 = nn.Conv2d(8, 16, kernel_size=5)
        self.linear_1 = nn.Linear(16 * (self.image_dimension - 8) ** 2,
                                  self.output_dimension)

    def forward(self, x):
        # Reshape into image
        x = x.view(self.batch_size, 1,
                   self.image_dimension,
                   self.image_dimension)

        # Pass through layers
        x1 = F.relu(self.conv_1(x))
        x2 = F.relu(self.conv_2(x1))

        # Flatten
        x2 = x2.view(self.batch_size, -1)
        x3 = self.linear_1(x2)

        return x3
