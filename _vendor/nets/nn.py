import math

import torch

from utils.util import make_anchors


def fuse_conv(conv, norm):
    fused_conv = torch.nn.Conv2d(conv.in_channels,
                                 conv.out_channels,
                                 kernel_size=conv.kernel_size,
                                 stride=conv.stride,
                                 padding=conv.padding,
                                 groups=conv.groups,
                                 bias=True).requires_grad_(False).to(conv.weight.device)

    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_norm = torch.diag(norm.weight.div(torch.sqrt(norm.eps + norm.running_var)))
    fused_conv.weight.copy_(torch.mm(w_norm, w_conv).view(fused_conv.weight.size()))

    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_norm = norm.bias - norm.weight.mul(norm.running_mean).div(torch.sqrt(norm.running_var + norm.eps))
    fused_conv.bias.copy_(torch.mm(w_norm, b_conv.reshape(-1, 1)).reshape(-1) + b_norm)

    return fused_conv


class Conv(torch.nn.Module):
    def __init__(self, in_ch, out_ch, activation, k=1, s=1, p=0, g=1):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_ch, out_ch, k, s, p, groups=g, bias=False)
        self.norm = torch.nn.BatchNorm2d(out_ch, eps=0.001, momentum=0.03)
        self.relu = activation

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))

    def fuse_forward(self, x):
        return self.relu(self.conv(x))


class Residual(torch.nn.Module):
    def __init__(self, ch, e=0.5):
        super().__init__()
        self.conv1 = Conv(ch, int(ch * e), torch.nn.SiLU(), k=3, p=1)
        self.conv2 = Conv(int(ch * e), ch, torch.nn.SiLU(), k=3, p=1)

    def forward(self, x):
        return x + self.conv2(self.conv1(x))


class IRDCB(torch.nn.Module):
    # Inverted Residual Depthwise Convolution Block
    # arXiv:2509.22365 - HierLight-YOLO
    def __init__(self, in_ch, out_ch, t=2):
        super().__init__()
        c_exp = int(in_ch * t)
        self.expand = Conv(in_ch, c_exp, torch.nn.SiLU())
        self.dw1 = Conv(c_exp, c_exp, torch.nn.SiLU(), k=3, p=1, g=c_exp)
        self.dw2 = Conv(c_exp, c_exp, torch.nn.SiLU(), k=3, p=1, g=c_exp)
        self.compress = Conv(c_exp, out_ch, torch.nn.SiLU())
        self.residual = (in_ch == out_ch)

    def forward(self, x):
        y = self.compress(self.dw2(self.dw1(self.expand(x))))
        return x + y if self.residual else y


class LDown(torch.nn.Module):
    # Lightweight Downsample module
    # arXiv:2509.22365 - HierLight-YOLO
    def __init__(self, in_ch, out_ch, k=3, s=2):
        super().__init__()
        p = k // 2
        self.dw = torch.nn.Conv2d(in_ch, in_ch, k, s, p, groups=in_ch, bias=False)
        self.dw_norm = torch.nn.BatchNorm2d(in_ch, eps=0.001, momentum=0.03)
        self.conv = Conv(in_ch, out_ch, torch.nn.SiLU())

    def forward(self, x):
        return self.conv(self.dw_norm(self.dw(x)))


class HFCC(torch.nn.Module):
    # Hierarchical Feature Channel Compression
    # arXiv:2509.22365 - HierLight-YOLO
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = Conv(in_ch, out_ch, torch.nn.SiLU())

    def forward(self, x):
        return self.conv(x)


class CSPModule(torch.nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = Conv(in_ch, out_ch // 2, torch.nn.SiLU())
        self.conv2 = Conv(in_ch, out_ch // 2, torch.nn.SiLU())
        self.conv3 = Conv(2 * (out_ch // 2), out_ch, torch.nn.SiLU())
        self.res_m = torch.nn.Sequential(Residual(out_ch // 2, e=1.0),
                                         Residual(out_ch // 2, e=1.0))

    def forward(self, x):
        y = self.res_m(self.conv1(x))
        return self.conv3(torch.cat((y, self.conv2(x)), dim=1))


class CSP(torch.nn.Module):
    def __init__(self, in_ch, out_ch, n, csp, r):
        super().__init__()
        self.conv1 = Conv(in_ch, 2 * (out_ch // r), torch.nn.SiLU())
        self.conv2 = Conv((2 + n) * (out_ch // r), out_ch, torch.nn.SiLU())

        if not csp:
            self.res_m = torch.nn.ModuleList(Residual(out_ch // r) for _ in range(n))
        else:
            self.res_m = torch.nn.ModuleList(CSPModule(out_ch // r, out_ch // r) for _ in range(n))

    def forward(self, x):
        y = list(self.conv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.res_m)
        return self.conv2(torch.cat(y, dim=1))


class SPP(torch.nn.Module):
    def __init__(self, in_ch, out_ch, k=5):
        super().__init__()
        self.conv1 = Conv(in_ch, in_ch // 2, torch.nn.SiLU())
        self.conv2 = Conv(in_ch * 2, out_ch, torch.nn.SiLU())
        self.res_m = torch.nn.MaxPool2d(k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.conv1(x)
        y1 = self.res_m(x)
        y2 = self.res_m(y1)
        return self.conv2(torch.cat(tensors=[x, y1, y2, self.res_m(y2)], dim=1))


class Attention(torch.nn.Module):

    def __init__(self, ch, num_head):
        super().__init__()
        self.num_head = num_head
        self.dim_head = ch // num_head
        self.dim_key = self.dim_head // 2
        self.scale = self.dim_key ** -0.5

        self.qkv = Conv(ch, ch + self.dim_key * num_head * 2, torch.nn.Identity())

        self.conv1 = Conv(ch, ch, torch.nn.Identity(), k=3, p=1, g=ch)
        self.conv2 = Conv(ch, ch, torch.nn.Identity())

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv(x)
        qkv = qkv.view(b, self.num_head, self.dim_key * 2 + self.dim_head, h * w)

        q, k, v = qkv.split([self.dim_key, self.dim_key, self.dim_head], dim=2)

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)

        x = (v @ attn.transpose(-2, -1)).view(b, c, h, w) + self.conv1(v.reshape(b, c, h, w))
        return self.conv2(x)


class PSABlock(torch.nn.Module):

    def __init__(self, ch, num_head):
        super().__init__()
        self.conv1 = Attention(ch, num_head)
        self.conv2 = torch.nn.Sequential(Conv(ch, ch * 2, torch.nn.SiLU()),
                                         Conv(ch * 2, ch, torch.nn.Identity()))

    def forward(self, x):
        x = x + self.conv1(x)
        return x + self.conv2(x)


class PSA(torch.nn.Module):
    def __init__(self, ch, n):
        super().__init__()
        self.conv1 = Conv(ch, 2 * (ch // 2), torch.nn.SiLU())
        self.conv2 = Conv(2 * (ch // 2), ch, torch.nn.SiLU())
        self.res_m = torch.nn.Sequential(*(PSABlock(ch // 2, ch // 128) for _ in range(n)))

    def forward(self, x):
        x, y = self.conv1(x).chunk(2, 1)
        return self.conv2(torch.cat(tensors=(x, self.res_m(y)), dim=1))


class DarkNet(torch.nn.Module):
    def __init__(self, width, depth, csp):
        super().__init__()
        self.p1 = []
        self.p2 = []
        self.p3 = []
        self.p4 = []
        self.p5 = []

        # p1/2
        self.p1.append(Conv(width[0], width[1], torch.nn.SiLU(), k=3, s=2, p=1))
        # p2/4
        self.p2.append(Conv(width[1], width[2], torch.nn.SiLU(), k=3, s=2, p=1))
        self.p2.append(CSP(width[2], width[3], depth[0], csp[0], r=4))
        # p3/8
        self.p3.append(Conv(width[3], width[3], torch.nn.SiLU(), k=3, s=2, p=1))
        self.p3.append(CSP(width[3], width[4], depth[1], csp[0], r=4))
        # p4/16
        self.p4.append(Conv(width[4], width[4], torch.nn.SiLU(), k=3, s=2, p=1))
        self.p4.append(CSP(width[4], width[4], depth[2], csp[1], r=2))
        # p5/32
        self.p5.append(Conv(width[4], width[5], torch.nn.SiLU(), k=3, s=2, p=1))
        self.p5.append(CSP(width[5], width[5], depth[3], csp[1], r=2))
        self.p5.append(SPP(width[5], width[5]))
        self.p5.append(PSA(width[5], depth[4]))

        self.p1 = torch.nn.Sequential(*self.p1)
        self.p2 = torch.nn.Sequential(*self.p2)
        self.p3 = torch.nn.Sequential(*self.p3)
        self.p4 = torch.nn.Sequential(*self.p4)
        self.p5 = torch.nn.Sequential(*self.p5)

    def forward(self, x):
        p1 = self.p1(x)
        p2 = self.p2(p1)
        p3 = self.p3(p2)
        p4 = self.p4(p3)
        p5 = self.p5(p4)
        return p3, p4, p5


class DarkFPN(torch.nn.Module):
    def __init__(self, width, depth, csp):
        super().__init__()
        self.up = torch.nn.Upsample(scale_factor=2)
        self.h1 = CSP(width[4] + width[5], width[4], depth[5], csp[0], r=2)
        self.h2 = CSP(width[4] + width[4], width[3], depth[5], csp[0], r=2)
        self.h3 = Conv(width[3], width[3], torch.nn.SiLU(), k=3, s=2, p=1)
        self.h4 = CSP(width[3] + width[4], width[4], depth[5], csp[0], r=2)
        self.h5 = Conv(width[4], width[4], torch.nn.SiLU(), k=3, s=2, p=1)
        self.h6 = CSP(width[4] + width[5], width[5], depth[5], csp[1], r=2)

    def forward(self, x):
        p3, p4, p5 = x
        p4 = self.h1(torch.cat(tensors=[self.up(p5), p4], dim=1))
        p3 = self.h2(torch.cat(tensors=[self.up(p4), p3], dim=1))
        p4 = self.h4(torch.cat(tensors=[self.h3(p3), p4], dim=1))
        p5 = self.h6(torch.cat(tensors=[self.h5(p4), p5], dim=1))
        return p3, p4, p5


class HEPAN(torch.nn.Module):
    # Hierarchical Extended Path Aggregation Network
    # arXiv:2509.22365 - HierLight-YOLO
    def __init__(self, width, n):
        super().__init__()
        self.up = torch.nn.Upsample(scale_factor=2)

        # HFCC at backbone outputs P2-P5
        self.hfcc2 = HFCC(width[3], width[3])
        self.hfcc3 = HFCC(width[4], width[3])
        self.hfcc4 = HFCC(width[4], width[3])
        self.hfcc5 = HFCC(width[5], width[3])

        # Top-down path (P5→P4→P3→P2)
        self.td4 = torch.nn.Sequential(*(IRDCB(width[3] * 2, width[3]) for _ in range(n)))
        self.td3 = torch.nn.Sequential(*(IRDCB(width[3] * 2, width[3]) for _ in range(n)))
        self.td2 = torch.nn.Sequential(*(IRDCB(width[3] * 2, width[3]) for _ in range(n)))

        # Bottom-up path (P2→P3→P4→P5)
        self.down_bu3 = LDown(width[3], width[3])
        self.bu3 = torch.nn.Sequential(*(IRDCB(width[3] * 2, width[3]) for _ in range(n)))
        self.down_bu4 = LDown(width[3], width[3])
        self.bu4 = torch.nn.Sequential(*(IRDCB(width[3] * 2, width[3]) for _ in range(n)))
        self.down_bu5 = LDown(width[3], width[3])
        self.bu5 = torch.nn.Sequential(*(IRDCB(width[3] * 2, width[3]) for _ in range(n)))

    def forward(self, x):
        p2, p3, p4, p5 = x
        # HFCC compression
        p2c = self.hfcc2(p2)
        p3c = self.hfcc3(p3)
        p4c = self.hfcc4(p4)
        p5c = self.hfcc5(p5)
        # Top-down (P5→P4→P3→P2)
        n4 = self.td4(torch.cat([self.up(p5c), p4c], dim=1))
        n3 = self.td3(torch.cat([self.up(n4), p3c], dim=1))
        n2 = self.td2(torch.cat([self.up(n3), p2c], dim=1))
        # Bottom-up (P2→P3→P4→P5) with dense skip connections
        f3 = self.bu3(torch.cat([self.down_bu3(n2), n3], dim=1))
        f4 = self.bu4(torch.cat([self.down_bu4(f3), n4], dim=1))
        f5 = self.bu5(torch.cat([self.down_bu5(f4), p5c], dim=1))
        return n2, f3, f4, f5


class DFL(torch.nn.Module):
    # Generalized Focal Loss
    # https://ieeexplore.ieee.org/document/9792391
    def __init__(self, ch=16):
        super().__init__()
        self.ch = ch
        self.conv = torch.nn.Conv2d(ch, out_channels=1, kernel_size=1, bias=False).requires_grad_(False)
        x = torch.arange(ch, dtype=torch.float).view(1, ch, 1, 1)
        self.conv.weight.data[:] = torch.nn.Parameter(x)

    def forward(self, x):
        b, c, a = x.shape
        x = x.view(b, 4, self.ch, a).transpose(2, 1)
        return self.conv(x.softmax(1)).view(b, 4, a)


class Head(torch.nn.Module):
    anchors = torch.empty(0)
    strides = torch.empty(0)

    def __init__(self, nc=80, filters=()):
        super().__init__()
        self.ch = 16  # DFL channels
        self.nc = nc  # number of classes
        self.nl = len(filters)  # number of detection layers
        self.no = nc + self.ch * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build

        box = max(64, filters[0] // 4)
        cls = max(80, filters[0], self.nc)

        self.dfl = DFL(self.ch)
        self.box = torch.nn.ModuleList(torch.nn.Sequential(Conv(x, box,torch.nn.SiLU(), k=3, p=1),
                                                           Conv(box, box,torch.nn.SiLU(), k=3, p=1),
                                                           torch.nn.Conv2d(box, out_channels=4 * self.ch,
                                                                           kernel_size=1)) for x in filters)
        self.cls = torch.nn.ModuleList(torch.nn.Sequential(Conv(x, x, torch.nn.SiLU(), k=3, p=1, g=x),
                                                           Conv(x, cls, torch.nn.SiLU()),
                                                           Conv(cls, cls, torch.nn.SiLU(), k=3, p=1, g=cls),
                                                           Conv(cls, cls, torch.nn.SiLU()),
                                                           torch.nn.Conv2d(cls, out_channels=self.nc,
                                                                           kernel_size=1)) for x in filters)

    def forward(self, x):
        for i, (box, cls) in enumerate(zip(self.box, self.cls)):
            x[i] = torch.cat(tensors=(box(x[i]), cls(x[i])), dim=1)
        if self.training:
            return x

        self.anchors, self.strides = (i.transpose(0, 1) for i in make_anchors(x, self.stride))
        x = torch.cat([i.view(x[0].shape[0], self.no, -1) for i in x], dim=2)
        box, cls = x.split(split_size=(4 * self.ch, self.nc), dim=1)

        a, b = self.dfl(box).chunk(2, 1)
        a = self.anchors.unsqueeze(0) - a
        b = self.anchors.unsqueeze(0) + b
        box = torch.cat(tensors=((a + b) / 2, b - a), dim=1)

        return torch.cat(tensors=(box * self.strides, cls.sigmoid()), dim=1)

    def initialize_biases(self):
        # Initialize biases
        # WARNING: requires stride availability
        for box, cls, s in zip(self.box, self.cls, self.stride):
            # box
            box[-1].bias.data[:] = 1.0
            # cls (.01 objects, 80 classes, 640 image)
            cls[-1].bias.data[:self.nc] = math.log(5 / self.nc / (640 / s) ** 2)


class YOLO(torch.nn.Module):
    def __init__(self, width, depth, csp, num_classes):
        super().__init__()
        self.net = DarkNet(width, depth, csp)
        self.fpn = DarkFPN(width, depth, csp)

        img_dummy = torch.zeros(1, width[0], 256, 256)
        self.head = Head(num_classes, (width[3], width[4], width[5]))
        self.head.stride = torch.tensor([256 / x.shape[-2] for x in self.forward(img_dummy)])
        self.stride = self.head.stride
        self.head.initialize_biases()

    def forward(self, x):
        x = self.net(x)
        x = self.fpn(x)
        return self.head(list(x))

    def fuse(self):
        for m in self.modules():
            if type(m) is Conv and hasattr(m, 'norm'):
                m.conv = fuse_conv(m.conv, m.norm)
                m.forward = m.fuse_forward
                delattr(m, 'norm')
        return self


def yolo_v11_n(num_classes: int = 80):
    csp = [False, True]
    depth = [1, 1, 1, 1, 1, 1]
    width = [3, 16, 32, 64, 128, 256]
    return YOLO(width, depth, csp, num_classes)


def yolo_v11_t(num_classes: int = 80):
    csp = [False, True]
    depth = [1, 1, 1, 1, 1, 1]
    width = [3, 24, 48, 96, 192, 384]
    return YOLO(width, depth, csp, num_classes)


def yolo_v11_s(num_classes: int = 80):
    csp = [False, True]
    depth = [1, 1, 1, 1, 1, 1]
    width = [3, 32, 64, 128, 256, 512]
    return YOLO(width, depth, csp, num_classes)


def yolo_v11_m(num_classes: int = 80):
    csp = [True, True]
    depth = [1, 1, 1, 1, 1, 1]
    width = [3, 64, 128, 256, 512, 512]
    return YOLO(width, depth, csp, num_classes)


def yolo_v11_l(num_classes: int = 80):
    csp = [True, True]
    depth = [2, 2, 2, 2, 2, 2]
    width = [3, 64, 128, 256, 512, 512]
    return YOLO(width, depth, csp, num_classes)


def yolo_v11_x(num_classes: int = 80):
    csp = [True, True]
    depth = [2, 2, 2, 2, 2, 2]
    width = [3, 96, 192, 384, 768, 768]
    return YOLO(width, depth, csp, num_classes)


class HierLightBackbone(torch.nn.Module):
    # Backbone using LDown + IRDCB, outputs P2-P5
    # arXiv:2509.22365 - HierLight-YOLO
    def __init__(self, width, depth):
        super().__init__()
        self.p1 = []
        self.p2 = []
        self.p3 = []
        self.p4 = []
        self.p5 = []

        # p1/2
        self.p1.append(Conv(width[0], width[1], torch.nn.SiLU(), k=3, s=2, p=1))
        # p2/4
        self.p2.append(LDown(width[1], width[2]))
        self.p2.append(IRDCB(width[2], width[3]))
        for _ in range(depth[0] - 1):
            self.p2.append(IRDCB(width[3], width[3]))
        # p3/8
        self.p3.append(LDown(width[3], width[3]))
        self.p3.append(IRDCB(width[3], width[4]))
        for _ in range(depth[1] - 1):
            self.p3.append(IRDCB(width[4], width[4]))
        # p4/16
        self.p4.append(LDown(width[4], width[4]))
        for _ in range(depth[2]):
            self.p4.append(IRDCB(width[4], width[4]))
        # p5/32
        self.p5.append(LDown(width[4], width[5]))
        for _ in range(depth[3]):
            self.p5.append(IRDCB(width[5], width[5]))
        self.p5.append(SPP(width[5], width[5]))

        self.p1 = torch.nn.Sequential(*self.p1)
        self.p2 = torch.nn.Sequential(*self.p2)
        self.p3 = torch.nn.Sequential(*self.p3)
        self.p4 = torch.nn.Sequential(*self.p4)
        self.p5 = torch.nn.Sequential(*self.p5)

    def forward(self, x):
        p1 = self.p1(x)
        p2 = self.p2(p1)
        p3 = self.p3(p2)
        p4 = self.p4(p3)
        p5 = self.p5(p4)
        return p2, p3, p4, p5


class HierLightHead(Head):
    # 4-scale detection head for HierLight-YOLO (P2, P3, P4, P5)
    # arXiv:2509.22365 - HierLight-YOLO
    def __init__(self, nc=80, filters=()):
        super().__init__(nc, filters)


class HierLightYOLO(torch.nn.Module):
    # arXiv:2509.22365 - HierLight-YOLO
    def __init__(self, width, depth, num_classes):
        super().__init__()
        self.net = HierLightBackbone(width, depth)
        self.fpn = HEPAN(width, depth[4])

        img_dummy = torch.zeros(1, width[0], 256, 256)
        self.head = HierLightHead(num_classes, (width[3], width[3], width[3], width[3]))
        self.head.stride = torch.tensor([256 / x.shape[-2] for x in self.forward(img_dummy)])
        self.stride = self.head.stride
        self.head.initialize_biases()

    def forward(self, x):
        x = self.net(x)
        x = self.fpn(x)
        return self.head(list(x))

    def fuse(self):
        for m in self.modules():
            if type(m) is Conv and hasattr(m, 'norm'):
                m.conv = fuse_conv(m.conv, m.norm)
                m.forward = m.fuse_forward
                delattr(m, 'norm')
        return self


def hierlight_yolo_n(num_classes: int = 10):
    depth = [1, 1, 1, 1, 1]
    width = [3, 16, 32, 64, 128, 256]
    return HierLightYOLO(width, depth, num_classes)


def hierlight_yolo_s(num_classes: int = 10):
    depth = [1, 1, 1, 1, 1]
    width = [3, 32, 64, 128, 256, 512]
    return HierLightYOLO(width, depth, num_classes)


def hierlight_yolo_m(num_classes: int = 10):
    depth = [2, 2, 2, 2, 1]
    width = [3, 48, 96, 192, 384, 576]
    return HierLightYOLO(width, depth, num_classes)


MODEL_REGISTRY = {
    'yolo_v11_n': yolo_v11_n,
    'yolo_v11_t': yolo_v11_t,
    'yolo_v11_s': yolo_v11_s,
    'yolo_v11_m': yolo_v11_m,
    'yolo_v11_l': yolo_v11_l,
    'yolo_v11_x': yolo_v11_x,
    'hierlight_yolo_n': hierlight_yolo_n,
    'hierlight_yolo_s': hierlight_yolo_s,
    'hierlight_yolo_m': hierlight_yolo_m,
}


def build_model(name, num_classes):
    if name not in MODEL_REGISTRY:
        raise ValueError(f'Unknown model: {name}. Available: {list(MODEL_REGISTRY.keys())}')
    return MODEL_REGISTRY[name](num_classes)
