import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out) * x

class MyChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=4):
        super(MyChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1_a   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc1_m   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)

        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc1_a(self.avg_pool(x))
        max_out = self.fc1_m(self.max_pool(x))
        out = avg_out + max_out
        out = self.relu1(out)
        out = self.sigmoid(self.fc2(out))
        return out * x

class MyChannelAttention_eca(nn.Module):
    def __init__(self, in_planes, ratio=4):
        super(MyChannelAttention_eca, self).__init__()
        k_size = 3
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv1d(1, in_planes // ratio, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.conv2 = nn.Conv1d(1, in_planes // ratio, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)

        self.conv3   = nn.Conv1d(in_planes // ratio, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.avg_pool(x)
        avg_out = self.conv1(avg_out.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        max_out = self.max_pool(x)
        max_out = self.conv2(max_out.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        out = avg_out + max_out
        out = self.relu1(out)
        out = self.conv3(out.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        out = self.sigmoid(out)
        return  out.expand_as(x) * x

class MyAttention_cs(nn.Module):
    def __init__(self, in_planes, ratio=4):
        super(MyAttention_cs, self).__init__()
        k_size = 3
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv1d(1, in_planes // ratio, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.conv2 = nn.Conv1d(1, in_planes // ratio, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)

        self.conv3   = nn.Conv1d(in_planes // ratio, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

        self.spatialAttention = SpatialAttention(kernel_size=3)

    def forward(self, x):
        avg_out = self.avg_pool(x)
        avg_out = self.conv1(avg_out.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        max_out = self.max_pool(x)
        max_out = self.conv2(max_out.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        out = avg_out + max_out
        out = self.relu1(out)
        out = self.conv3(out.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        out = self.sigmoid(out)
        out_sp = self.spatialAttention(x)

        return  out.expand_as(x) * x + x + out_sp

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv1(y)
        return self.sigmoid(y) * x


class MyAtte_block(nn.Module):
    def __init__(self, in_planes, ratio=4):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes,in_planes//ratio,kernel_size=3,padding=1,bias=False)
        self.relu1 = nn.ReLU()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.conv2 = nn.Conv2d(in_planes//ratio,1,kernel_size=3,padding=1,bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out1 = self.conv1(self.relu1(x))
        ca_weight = self.sigmoid(self.fc(self.avg_pool(out1)))
        sa_weight = self.sigmoid(self.conv2(out1))

        y = sa_weight * (ca_weight * x)
        return y
class DCA(nn.Module):
    def __init__(self, in_planes, ratio=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.fc3   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu2 = nn.ReLU()
        self.fc4   = nn.Conv2d((in_planes // ratio)*2, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        tmp1 = self.relu1(self.fc1(self.avg_pool(x)))
        cs = self.sigmoid(self.fc2(tmp1)) * x

        tmp2 = self.relu2(self.fc3(self.avg_pool(cs)))
        tmp = torch.cat([tmp1,tmp2],dim=1)
        res = self.sigmoid(self.fc4(tmp)) * cs
        return res


if __name__ == "__main__":
    bs, c, h, w = 1, 32, 16, 16
    in_tensor = torch.ones(bs, c, h, w)

    net = MyChannelAttention_eca(32)
    print("in shape:",in_tensor.size())
    out_tensor = net(in_tensor)
    print("out shape:", out_tensor.size())