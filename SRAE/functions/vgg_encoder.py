import torch
import torch.nn as nn
from torchvision.models import vgg19

class block1(nn.Module):              
    def __init__(self):
        super(block1, self).__init__()
        vgg19_model = vgg19(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:4])

    def forward(self, x):
        return self.feature_extractor(x)

class block2(nn.Module):           
    def __init__(self):
        super(block2, self).__init__()
        vgg19_model = vgg19(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:9])

    def forward(self, x):
        return self.feature_extractor(x)

class block3(nn.Module):            
    def __init__(self):
        super(block3, self).__init__()
        vgg19_model = vgg19(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:18])

    def forward(self, x):
        return self.feature_extractor(x)

class block4(nn.Module):                  
    def __init__(self):
        super(block4, self).__init__()
        vgg19_model = vgg19(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:27])

    def forward(self, x):
        return self.feature_extractor(x)

class block5(nn.Module):                 
    def __init__(self):
        super(block5, self).__init__()
        vgg19_model = vgg19(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:36])

    def forward(self, x):
        return self.feature_extractor(x)