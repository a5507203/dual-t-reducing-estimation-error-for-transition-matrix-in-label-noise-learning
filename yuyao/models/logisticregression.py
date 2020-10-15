# Import all the packages
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f

__all__ = ['LogisticRegressionModel']

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_size=10, num_classes=1):
        super(LogisticRegressionModel, self).__init__()


        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        y_predict = f.sigmoid(self.linear(x)) # to convert to 1 or 0 
        return y_predict

# model = LogisticRegressionModel(1,1)
# criteria = nn.BCELoss()
# optimizer = optim.SGD(model.parameters(), 0.01)