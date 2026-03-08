# yolov5/models/custom/bifpn_lite.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SeparableConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, 3, 1, 1, groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)

class BiFPNLayer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # weights for fast normalized fusion (learnable)
        self.w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.eps = 1e-4
        self.conv_td = SeparableConvBlock(channels, channels)
        self.conv_bu = SeparableConvBlock(channels, channels)

    def forward(self, p3, p4, p5):
        # top-down
        w1 = F.relu(self.w1)
        w1 = w1 / (torch.sum(w1) + self.eps)
        p5_ups = F.interpolate(p5, scale_factor=2.0, mode='nearest')
        td = self.conv_td(w1[0]*p4 + w1[1]*p5_ups)

        # bottom-up
        w2 = F.relu(self.w2)
        w2 = w2 / (torch.sum(w2) + self.eps)
        td_ups = F.interpolate(td, scale_factor=2.0, mode='nearest')
        p3_out = self.conv_bu(w2[0]*p3 + w2[1]*td_ups + w2[2]*p3)  # simplified

        # return (p3_out, td, p5) as new pyramid
        return p3_out, td, p5

class BiFPN_Lite(nn.Module):
    def __init__(self, in_channels=[128,256,512], out_channels=256, layers=2):
        super().__init__()
        # unify channel sizes
        self.project_p3 = nn.Conv2d(in_channels[0], out_channels, 1,1,0)
        self.project_p4 = nn.Conv2d(in_channels[1], out_channels, 1,1,0)
        self.project_p5 = nn.Conv2d(in_channels[2], out_channels, 1,1,0)
        self.layers = nn.ModuleList([BiFPNLayer(out_channels) for _ in range(layers)])

    def forward(self, p3, p4, p5):
        p3 = self.project_p3(p3)
        p4 = self.project_p4(p4)
        p5 = self.project_p5(p5)
        for layer in self.layers:
            p3, p4, p5 = layer(p3,p4,p5)
        # return outputs in descending stride order expected by YOLO head: P3 (high-res), P4, P5
        return p3, p4, p5

if __name__ == "__main__":
    m = BiFPN_Lite()
    import torch
    x1 = torch.randn(2,128,40,40)
    x2 = torch.randn(2,256,20,20)
    x3 = torch.randn(2,512,10,10)
    y1,y2,y3 = m(x1,x2,x3)
    print(y1.shape, y2.shape, y3.shape)
