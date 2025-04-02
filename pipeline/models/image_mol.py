import torch
import torchvision
import math
import os
import logging
import torch.nn as nn


def get_support_model_names():
    return ["ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152"]


def load_model(modelname="ResNet18", imageSize=224, num_classes=2):
    assert modelname in get_support_model_names()
    if modelname == "ResNet18":
        model = torchvision.models.resnet18(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif modelname == "ResNet34":
        model = torchvision.models.resnet34(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif modelname == "ResNet50":
        model = torchvision.models.resnet50(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif modelname == "ResNet101":
        model = torchvision.models.resnet101(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif modelname == "ResNet152":
        model = torchvision.models.resnet152(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    else:
        raise Exception("{} is undefined".format(modelname))
    # emb_dim = model.fc.in_features
    return model


class ImageMol(nn.Module):
    def __init__(self, baseModel="ResNet18", jigsaw_classes=101, label1_classes=100, label2_classes=1000, label3_classes=10000):
        super(ImageMol, self).__init__()

        assert baseModel in get_support_model_names()

        self.baseModel = baseModel

        self.embedding_layer = nn.Sequential(*list(load_model(baseModel).children())[:-1])
        self.emb_dim = 512
        # self.bn = nn.BatchNorm1d(512)

        # self.jigsaw_classifier = nn.Linear(512, jigsaw_classes)
        # self.class_classifier1 = nn.Linear(512, label1_classes)
        # self.class_classifier2 = nn.Linear(512, label2_classes)
        # self.class_classifier3 = nn.Linear(512, label3_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.embedding_layer(x)
        x = x.view(x.size(0), -1)
        return x

        # x1 = self.jigsaw_classifier(x)
        # x2 = self.class_classifier1(x)
        # x3 = self.class_classifier2(x)
        # x4 = self.class_classifier3(x)

        # return x, x1, x2, x3, x4

    def load_from_pretrained(self, url_or_filename):
        if os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location="cpu")
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        msg = self.load_state_dict(state_dict, strict=False)

        print("Loaded from {}: {}".format(url_or_filename, msg))

        return msg

if __name__ == "__main__":
    ckpt = torch.load("ckpt/ImageMol.pth.tar", map_location="cpu")
    net = ImageMol("ResNet18")
    msg = net.load_state_dict(ckpt["state_dict"], strict=False)
    print(msg)
    x = torch.rand(2, 3, 224, 224)
    y = net(x)
    print(y.shape)
