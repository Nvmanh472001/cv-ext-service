import os, sys
import torch.nn as nn

from model.det_model.modules import MobileNetV3, RSEFPN, DBHead

class DetectionModel(nn.Module):
    def __init__(self, config, **kwargs):
        """
        the module for OCR.
        args:
            config (dict): the super parameters for module.
        """
        super(DetectionModel, self).__init__()

        in_channels = config.get('in_channels', 3)
        
        #build backbone
        config["Backbone"]['in_channels'] = in_channels
        self.backbone = MobileNetV3.build(config["Backbone"])
        in_channels = self.backbone.out_channels

        # build neck
        config['Neck']['in_channels'] = in_channels
        self.neck = RSEFPN.build(config['Neck'])
        in_channels = self.neck.out_channels

        config["Head"]['in_channels'] = in_channels
        self.head = DBHead.build(config["Head"], **kwargs)

        self._initialize_weights()

    def _initialize_weights(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


    def forward(self, x):
        y = dict()
        x = self.backbone(x)
        y["backbone_out"] = x
        if self.use_neck:
            x = self.neck(x)
        y["neck_out"] = x
        if self.use_head:
            x = self.head(x)
        return x