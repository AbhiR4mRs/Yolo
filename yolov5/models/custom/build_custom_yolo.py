# yolov5/models/custom/build_custom_yolo.py
import torch
import torch.nn as nn
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))  # adjust path to yolov5 root
from yolov5.models.custom.mobilenetv3_catr_backbone import MobileNetV3_CATr_Backbone
from yolov5.models.custom.bifpn_lite import BiFPN_Lite

# Attempt to import YOLO head from yolov5 repo
try:
    from yolov5.models.yolo import Detect
    from yolov5.models.common import Conv
except Exception as e:
    print("Make sure you run this from the root of the YOLOv5 repo (so 'yolov5' package is importable).", e)
    raise

class CustomYOLOv5(nn.Module):
    def __init__(self, nc=3, anchors=None):
        super().__init__()
        self.backbone = MobileNetV3_CATr_Backbone(pretrained=True)
        self.bifpn = BiFPN_Lite(in_channels=[128,256,512], out_channels=256, layers=2)
        # create small convs before detect head to produce required channels
        self.prep_p3 = Conv(256, 256, k=3, s=1)
        self.prep_p4 = Conv(256, 256, k=3, s=1)
        self.prep_p5 = Conv(256, 256, k=3, s=1)
        # anchors default to common YOLOv5 anchors if not provided
        if anchors is None:
            anchors = [[10,13, 16,30, 33,23], [30,61, 62,45, 59,119], [116,90, 156,198, 373,326]]
        # Detect expects channel list: e.g., [256, 256, 256]
        self.detect = Detect(nc=nc, anchors=anchors, ch=[256,256,256])

    def forward(self, x):
        p3, p4, p5 = self.backbone(x)          # backbone outputs
        f3, f4, f5 = self.bifpn(p3,p4,p5)      # fused features
        f3 = self.prep_p3(f3)
        f4 = self.prep_p4(f4)
        f5 = self.prep_p5(f5)
        out = self.detect([f3,f4,f5])
        return out

if __name__ == "__main__":
    # quick sanity test
    m = CustomYOLOv5(nc=3)
    x = torch.randn(1,3,320,320)
    y = m(x)
    # y is list of predictions per scale, each shape (bs, anchors, grid_h, grid_w, (5+nc))
    for i, t in enumerate(y):
        print(f"scale {i} ->", t.shape)
