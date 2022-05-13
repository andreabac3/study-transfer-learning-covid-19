from typing import *

import torch.nn as nn
from torchvision import models


def get_pretrained_vgg16(n_classes: int) -> nn.Module:
    model = models.vgg16(pretrained=True)

    # Freeze the encoder layers
    model = _freeze_parameters(model)
    # create the new last layer, it's size depends on the task addressed
    n_inputs = model.classifier[6].in_features

    # Add on classifier
    model.classifier[6] = nn.Sequential(
        nn.Linear(n_inputs, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, n_classes),
        nn.LogSoftmax(dim=1),
    )

    return model


def get_pretrained_resnet50(n_classes: int) -> nn.Module:
    model = models.resnet50(pretrained=True)
    model = _freeze_parameters(model)

    n_inputs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(n_inputs, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, n_classes),
        nn.LogSoftmax(dim=1),
    )

    return model


def get_pretrained_squeezenet(n_classes: int) -> nn.Module:
    model = models.squeezenet1_0(pretrained=True)
    model = _freeze_parameters(model)

    model.classifier[1] = nn.Conv2d(512, n_classes, kernel_size=(1, 1), stride=(1, 1))

    return model


def get_pretrained_alexnet(n_classes: int) -> nn.Module:
    model = models.alexnet(pretrained=True)
    model = _freeze_parameters(model)
    # create the new last layer, it's size depends on the task addressed
    n_inputs = model.classifier[6].in_features

    # Add on classifier
    model.classifier[6] = nn.Sequential(
        nn.Linear(n_inputs, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, n_classes),
        nn.LogSoftmax(dim=1),
    )
    return model


def get_pretrained_densenet(n_classes: int) -> nn.Module:
    model = models.densenet161(pretrained=True)
    model = _freeze_parameters(model)
    n_inputs = model.classifier.in_features

    # Add on classifier
    model.classifier = nn.Sequential(
        nn.Linear(n_inputs, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, n_classes),
        nn.LogSoftmax(dim=1),
    )
    return model


def get_pretrained_inception(n_classes: int) -> nn.Module:
    model = models.inception_v3(pretrained=True)
    model = _freeze_parameters(model)
    n_inputs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(n_inputs, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, n_classes),
        nn.LogSoftmax(dim=1),
    )

    return model


def get_pretrained_googlenet(n_classes: int) -> nn.Module:
    model = models.googlenet(pretrained=True)
    model = _freeze_parameters(model)
    n_inputs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(n_inputs, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, n_classes),
        nn.LogSoftmax(dim=1),
    )

    return model


def get_pretrained_shufflenet(n_classes: int) -> nn.Module:
    model = models.shufflenet_v2_x1_0(pretrained=True)
    model = _freeze_parameters(model)
    n_inputs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(n_inputs, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, n_classes),
        nn.LogSoftmax(dim=1),
    )
    return model


def _freeze_parameters(model: nn.Module) -> nn.Module:
    for param in model.parameters():
        param.requires_grad = False
    return model


def get_pretrained_mobilenet_v2(n_classes: int) -> nn.Module:
    model = models.mobilenet_v2(pretrained=True)
    model = _freeze_parameters(model)

    n_inputs = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Linear(n_inputs, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, n_classes),
        nn.LogSoftmax(dim=1),
    )

    return model


def get_pretrained_mobilenet_v3(n_classes: int) -> nn.Module:
    model = models.mobilenet_v3_large(pretrained=True)
    model = _freeze_parameters(model)

    n_inputs = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(n_inputs, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, n_classes),
        nn.LogSoftmax(dim=1),
    )
    return model


def get_pretrained_wide_resnet50(n_classes: int) -> nn.Module:
    model = models.wide_resnet50_2(pretrained=True)

    model = _freeze_parameters(model)

    n_inputs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(n_inputs, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, n_classes),
        nn.LogSoftmax(dim=1),
    )

    return model


def get_pretrained_mnasnet(n_classes: int) -> nn.Module:
    model = models.mnasnet1_0(pretrained=True)
    model = _freeze_parameters(model)

    n_inputs = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Linear(n_inputs, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, n_classes),
        nn.LogSoftmax(dim=1),
    )

    return model


def get_pretrained_resnext(n_classes: int) -> nn.Module:
    model = models.resnext50_32x4d(pretrained=True)

    model = _freeze_parameters(model)
    n_inputs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(n_inputs, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, n_classes),
        nn.LogSoftmax(dim=1),
    )

    return model


def get_pretrained(pretrained_name: str, n_classes: int) -> nn.Module:
    pretrained_dict: dict = {
        "vgg16": get_pretrained_vgg16,
        "resnet50": get_pretrained_resnet50,
        "squeezenet": get_pretrained_squeezenet,
        "alexnet": get_pretrained_alexnet,
        "densenet": get_pretrained_densenet,
        "inception": get_pretrained_inception,
        "googlenet": get_pretrained_googlenet,
        "shufflenet": get_pretrained_shufflenet,
        "mobilenet_v2": get_pretrained_mobilenet_v2,
        "mobilenet_v3": get_pretrained_mobilenet_v3,
        "wide_resnet50": get_pretrained_wide_resnet50,
        "mnasnet": get_pretrained_mnasnet,
        "resnext": get_pretrained_resnext,
    }
    return pretrained_dict[pretrained_name](n_classes)
