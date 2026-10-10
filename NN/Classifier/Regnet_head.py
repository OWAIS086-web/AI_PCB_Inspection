
from transformers import  RegNetForImageClassification, RegNetConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.Classifier.fpn import FPN, conv_bn_relu


class RegNetWithFPN(nn.Module):
    def __init__(self, regnet_name='facebook/regnet-y-040', head_conv=256 , device = "cuda"):
        super(RegNetWithFPN, self).__init__()
        self.device = device

        # Load the RegNet backbone
        config = RegNetConfig.from_pretrained(regnet_name, output_hidden_states=True)
        config.problem_type = "single_label_classification"
        config.num_labels = 5
        self.backbone = RegNetForImageClassification(config=config).to(self.device)
        # self.backbone.load_state_dict(torch.load('./models/Regnet_model/regnet_backbone.tar'))
        self.backbone.to(self.device).eval()
        


        # Define the FPN neck
        # Assuming backbone output feature maps have specific channels:
        self.fpn = FPN(in_channels=[128, 192, 512,1088],
            out_channels=64,
            num_outs=4,
            attention=False).to(self.device)
        

        self.feat_combine = torch.nn.Sequential(conv_bn_relu(64 * 4, 64, 3, padding=1, dilation=1),torch.nn.Conv2d(64, 64, 1),).to(self.device)

        self.feat_refine = torch.nn.Sequential(conv_bn_relu(64, 64, 3, padding=1, dilation=1)).to(self.device)

        self.heads = {'missing': 2, 'shifted': 2, 'ic_direction': 3, 'bridge': 3, 'capacitor_color': 4}
        
        for head in self.heads:
            classes = self.heads[head]
            fc = nn.Sequential(
                nn.Conv2d(64, head_conv, kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_conv, 32, kernel_size=3, stride=2, padding=1, bias=True),  # Second conv layer
                nn.ReLU(inplace=True),
                nn.Flatten(),
                nn.Linear(in_features=16384,out_features=classes),
                # nn.Sigmoid()
                ).to(self.device)
            self.__setattr__(head, fc)

    def forward_for_squeeze(self, features):
        # Feature squeeze and concat
        x1 = features[0]

        x2 = features[1]
        x2 = torch.nn.functional.interpolate(x2, scale_factor=2, mode='bilinear')

        x3 = features[2]
        x3 = torch.nn.functional.interpolate(x3, scale_factor=4, mode='bilinear')

        x4 = features[3]
        x4 = torch.nn.functional.interpolate(x4, scale_factor=8, mode='bilinear')

        x_concat = torch.cat([x1, x2, x3,x4], dim=1)
        x5 = self.feat_combine(x_concat)
        img_feat = self.feat_refine(x5)

        return img_feat



    def forward(self, x):
        # Forward pass through RegNet backbone
        x = self.backbone(x,output_hidden_states=True).hidden_states
        x = self.fpn(x[1:5])   # Feature maps of regnet to FPN
        x = self.forward_for_squeeze(x)  # COncating each layer feature maps
        ret = {}
        for head in self.heads:
            ret[head] = self.__getattr__(head)(x)
        return ret