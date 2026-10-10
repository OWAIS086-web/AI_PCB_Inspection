from transformers import RegNetForImageClassification, RegNetConfig, ResNetModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from NN.Classifier.neck import FPN, conv_bn_relu


class MultiHeadComponentClassifier(nn.Module):
    def __init__(self, config=None, logger=None):
        super(MultiHeadComponentClassifier, self).__init__()

        self.logger = logger

        if config is None:
            import json

            config = json.load(open("models/Classifier/model_config.json", "r"))

        self.config = config
        self.device = config["device"]

        # self.neck_config = self.config["neck"]
        self.head_config = self.config["heads"]
        self.backbone_config = self.config["backbone"]

        # self.regnet_cfg = RegNetConfig.from_pretrained(
        #     self.backbone_config["regnet_name"],
        #     output_hidden_states=self.backbone_config["output_hidden_states"],
        # )
        # self.backbone = RegNetForImageClassification.from_pretrained(
        #     self.backbone_config["regnet_name"],
        #     config=self.regnet_cfg,
        #     cache_dir=config["cache_dir"],
        # )
        # self.backbone.to(self.device)
        self.model = ResNetModel.from_pretrained("microsoft/resnet-18")

        # self.neck = FPN(
        #     in_channels=self.neck_config["in_channels"],
        #     out_channels=self.neck_config["out_channels"],
        #     num_outs=self.neck_config["num_outs"],
        #     attention=self.neck_config["attention"],
        # ).to(self.device)
        # self.neck.to(self.device)

        # self.combine_features = torch.nn.Sequential(
        #     conv_bn_relu(
        #         self.neck_config["out_channels"]
        #         * self.config["upsample"]["combine_upsample"],
        #         64,
        #         3,
        #         padding=1,
        #         dilation=1,
        #     ),
        #     torch.nn.Conv2d(64, 64, 1),
        # ).to(self.device)

        # self.refine_features = torch.nn.Sequential(
        #     conv_bn_relu(64, 64, 3, padding=1, dilation=1)
        # ).to(self.device)

        self.heads = self.head_config["outs"]

        for head in self.heads:
            classes = self.heads[head]
            fc = nn.Sequential(
                nn.Conv2d(
                    256,
                    self.head_config["conv_dim"],
                    kernel_size=3,
                    padding=1,
                    bias=True,
                ),
                nn.ReLU(),
                nn.Conv2d(
                    self.head_config["conv_dim"],
                    32,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=True,
                ),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(in_features=1024, out_features=classes),
                nn.Softmax(dim=1),
            ).to(self.device)
            self.__setattr__(head, fc)

    def concat_and_refine(self, features):
        x1 = features[0]
        x2 = features[1]
        x2 = torch.nn.functional.interpolate(x2, scale_factor=2, mode="bilinear")
        x3 = features[2]
        x3 = torch.nn.functional.interpolate(x3, scale_factor=4, mode="bilinear")
        x_concat = torch.cat([x1, x2, x3], dim=1)
        x4 = self.combine_features(x_concat)
        img_feat = self.refine_features(x4)

        return img_feat

    def forward(self, x):
        x = self.model(
            x, output_hidden_states=self.backbone_config["output_hidden_states"]
        ).hidden_states
        # x = self.neck(x[1:4])
        # x = self.concat_and_refine(x)
        ret = {}
        for head in self.heads:
            ret[head] = self.__getattr__(head)(x[3])
        return ret
