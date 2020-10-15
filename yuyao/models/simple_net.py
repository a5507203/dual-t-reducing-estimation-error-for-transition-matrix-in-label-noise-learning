 
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['Lenet','naivenet']




class NaiveNet(nn.Module):

    def __init__(self, input_dim=2, hidden_dim=25, output_dim=2):
        super(NaiveNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        out = F.relu(self.bn1(self.fc1(x)))
        out = F.relu(self.bn2(self.fc2(out)))
        out = self.fc3(out)
        return out



class Lenet(nn.Module):
    def __init__(self,num_classes=10,pretrained=False, input_channel=1):
        super(Lenet, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.dropout1 = nn.Dropout2d(0.5)
        self.dropout2 = nn.Dropout2d(0.5)
        self.T_revision = nn.Linear(num_classes, num_classes, False)

    def forward(self, x, revision=False):
        correction = self.T_revision.weight
        batch_size = x.shape[0]
        x = x.view(batch_size,-1)
        # print(x.shape[0])
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)    
        x = self.fc3(x) 

        if revision == True:
            return x, correction
        else:
            return x


def naivenet(input_dim=10,num_classes=2):
    return NaiveNet(input_dim=input_dim,output_dim=num_classes)


