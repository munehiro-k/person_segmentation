import segmentation_models_pytorch as smp
import torch.nn as nn


class SegmentationNet(nn.Module):
    def __init__(self):
        super(SegmentationNet, self).__init__()
        net = smp.DeepLabV3Plus(
            encoder_name="timm-mobilenetv3_small_minimal_100",
            encoder_weights='imagenet',
            encoder_depth=3,
            encoder_output_stride=16,
            upsampling=2,
            classes=1,
            activation=None)
        self.params_to_update = []

        self.input_bn = nn.BatchNorm2d(3)
        self.net = net

    def forward(self, x):
        return self.net(self.input_bn(x))
