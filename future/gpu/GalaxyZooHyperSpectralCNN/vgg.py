import math
import torch.nn as nn

__all__ = ['vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn']

class GalaxyZooVGG(nn.Module):
    def __init__(self, conv_layers, num_classes=6):
        super(GalaxyZooVGG, self).__init__()
        self.conv_layers = conv_layers
        self.classifier = nn.Sequential(
                            nn.Dropout(),
                            nn.Conv2d(in_channels=512, out_channels=100, kernel_size=1),
                            #nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1),
                            nn.Conv2d(in_channels=100, out_channels=num_classes, kernel_size=1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.classifier(x)
        x = x.view([1, 6])
        return x
config = {
    'vgg11': [64, 'pool', 128, 'pool', 256, 256, 'pool', 512, 512, 'pool', 512, 512, 'adapool'],
    'vgg11_nopool': [64, 'pool', 128, 'adapool'],
    'vgg13': [64, 64, 'pool', 128, 128, 'pool', 256, 256, 'pool', 512, 512, 'pool', 512, 512, 'adapool'],
    'vgg16': [64, 64, 'pool', 128, 128, 'pool', 256, 256, 256, 'pool', 512, 512, 512, 'pool', 512, 512, 512, 'adapool'],
    'vgg19': [64, 64, 'pool', 128, 128, 'pool', 256, 256, 256, 256, 'pool', 512, 512, 512, 512, 'pool', 512, 512, 512, 512, 'adapool'],
}

def make_layers(config, in_channels=5, batch_norm=False, pooling_type='max'):
    layers = []
    for layer_dim in config:
        if layer_dim == 'pool':
            if pooling_type == 'max':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif pooling_type == 'avg':
                layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
            else:
                raise ValueError("pooling_type must be 'max' or 'avg'")
        elif layer_dim == 'adapool':
            if pooling_type == 'max':
                layers += [nn.AdaptiveMaxPool2d(1)]
            elif pooling_type == 'avg':
                layers += [nn.AdaptiveAvgPool2d(1)]
            else:
                raise ValueError("pooling_type must be 'max' or 'avg'")
        else:
            conv2d = nn.Conv2d(in_channels, layer_dim, kernel_size=3, padding=1, stride=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(layer_dim), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = layer_dim
    return nn.Sequential(*layers)

def vgg11(in_channels=5):
    return GalaxyZooVGG(make_layers(config['vgg11'], in_channels=in_channels))

def vgg11_bn(in_channels=5):
    return GalaxyZooVGG(make_layers(config['vgg11'], in_channels=in_channels, batch_norm=True))

def vgg11_nopool(in_channels=5):
    return GalaxyZooVGG(make_layers(config['vgg11_nopool'], in_channels=in_channels))

def vgg13(in_channels=5):
    return GalaxyZooVGG(make_layers(config['vgg13'], in_channels=in_channels))

def vgg13_bn(in_channels=5):
    return GalaxyZooVGG(make_layers(config['vgg13'], in_channels=in_channels, batch_norm=True))

def vgg16(in_channels=5):
    return GalaxyZooVGG(make_layers(config['vgg16'], in_channels=in_channels))

def vgg16_bn(in_channels=5):
    return GalaxyZooVGG(make_layers(config['vgg16'], in_channels=in_channels, batch_norm=True))

def vgg19(in_channels=5):
    return GalaxyZooVGG(make_layers(config['vgg19'], in_channels=in_channels))

def vgg19_bn(in_channels=5):
    return GalaxyZooVGG(make_layers(config['vgg19'], in_channels=in_channels, batch_norm=True))


