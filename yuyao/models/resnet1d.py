import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F


__all__= ["TR_Resnet1d18","TR_Resnet1d18lr"]
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)



class BasicBlock3x3(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock3x3, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class MSResNet(nn.Module):
    inplanes = 20
    def __init__(self, input_channel=1, layers=[1, 1, 1, 1], num_classes=2):
        super(MSResNet, self).__init__()
        self.conv1 = nn.Conv1d(input_channel, 20, kernel_size=2, stride=2, padding=3,  bias=False)
        self.bn1 = nn.BatchNorm1d(20)
        self.layer1 = self._make_layer(BasicBlock3x3, 20, layers[0], stride=2)
        self.layer2 = self._make_layer(BasicBlock3x3, 40, layers[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock3x3, 80, layers[2], stride=2)
        self.layer4 = self._make_layer(BasicBlock3x3, 160, layers[3], stride=2)
        self.avgpool1d = nn.AdaptiveAvgPool1d((1))
        self.fc = nn.Linear(160, num_classes)
        self.T_revision = nn.Linear(num_classes, num_classes, False)

    def _make_layer(self, block, planes, blocks, stride=2):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, revision=False):
        correction = self.T_revision.weight
        x = x.unsqueeze(1)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool1d(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        if revision == True:
            return out, correction
        else:
            return out


class MSResNetlr(nn.Module):
    inplanes = 20
    def __init__(self, input_channel=1, layers=[1, 1, 1, 1], num_classes=2):
        super(MSResNetlr, self).__init__()
        self.conv1 = nn.Conv1d(input_channel, 20, kernel_size=2, stride=2, padding=3,  bias=False)
        self.bn1 = nn.BatchNorm1d(20)
        self.layer1 = self._make_layer(BasicBlock3x3, 20, layers[0], stride=2)
        self.layer2 = self._make_layer(BasicBlock3x3, 40, layers[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock3x3, 80, layers[2], stride=2)
        self.layer4 = self._make_layer(BasicBlock3x3, 160, layers[3], stride=2)
        self.avgpool1d = nn.AdaptiveAvgPool1d((1))
        self.fc = nn.Linear(160, num_classes)
        self.T_revision = nn.Linear(num_classes, num_classes, False)

    def _make_layer(self, block, planes, blocks, stride=2):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, revision=False):
        correction = self.T_revision.weight
        x = x.unsqueeze(1)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool1d(out)
        out = out.view(out.size(0), -1)
        out = F.sigmoid(self.fc(out))

        if revision == True:
            return out, correction
        else:
            return out



def TR_Resnet1d18(num_classes=2):
   return MSResNet(layers=[2, 2, 2, 2],num_classes=num_classes)


def TR_Resnet1d18lr(num_classes=2):
   return MSResNet(layers=[2, 2, 2, 2],num_classes=1)