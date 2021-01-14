from dropblock import DropBlock2D
import torch
import torch.nn.functional as F
import torch.nn as nn

class CBR(nn.Module):
    def __init__(self, in_channel,out_channel,kernel_size=3,padding=1,stride=1,bias=True):
        self.conv = nn.Conv2d(in_channel,out_channel,kernel_size,padding,stride,bias)
        self.bn = nn.BatchNorm2d(out_channel)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.Relu(x)

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

class AC_Attention(nn.Module):
    def __init__(self, in_channel):
        super(AC_Attention, self).__init__()
        self.block1x3 = nn.Sequential(nn.Conv2d(in_channel, 1, (1,5), stride=1, padding=(0,2)),
                                        )
        self.block3x1 = nn.Sequential(nn.Conv2d(in_channel, 1, (5,1), stride=1,padding=(2,0)),
                                        )
        self.block3x3 = nn.Sequential(nn.Conv2d(in_channel, 1, (3,3),stride=1,padding=(1,1)),
                                        )

        self.conv_final = nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out1 = self.block1x3(x)
        out2 = self.block3x1(x)
        out3 = self.block3x3(x)
        out = torch.cat([out1, out2, out3],dim=1)
        # out = F.relu(out)
        out = self.sigmoid(self.conv_final(out))
        return out * x

class Eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3,dilat=1):
        super(Eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.conv2 = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        # b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv1(y.squeeze(-1).transpose(-1, -2))
        y = F.relu(y)
        y = self.conv2(y).transpose(-1, -2).unsqueeze(-1)
        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

class MECA(nn.Module):
    def __init__(self, channel):
        super(MECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        k = [2,4,8]
        k_size = []
        for i in k:
            k_size.append(channel//i + 1)
        self.conv1 = nn.Conv1d(2, 1, kernel_size=k_size[0], padding=(k_size[0] - 1) // 2, bias=False)
        self.conv2 = nn.Conv1d(2, 1, kernel_size=k_size[1], padding=(k_size[1] - 1) // 2, bias=False)
        self.conv3 = nn.Conv1d(2, 1, kernel_size=k_size[2], padding=(k_size[2] - 1) // 2, bias=False)
        # print("*****************", k_size)
        #self.conv_final = nn.C#onv1d(3, 1, kernel_size=k_size[3], padding=(k_size[3] - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        # b, c, h, w = x.size()
        # print(x.size())
        # feature descriptor on the global spatial information
        y1 = self.avg_pool(x).squeeze(-1).transpose(-1, -2)
        y2 = self.max_pool(x).squeeze(-1).transpose(-1, -2)
        y = torch.cat([y1,y2],dim=1)

        # Two different branches of ECA module
        y = self.conv1(y)+self.conv2(y)+self.conv3(y)
        # y = F.relu(y)
        # y = self.conv_final(y).transpose(-1, -2).unsqueeze(-1)
        # Multi-scale information fusion
        y = self.sigmoid(y.transpose(-1, -2).unsqueeze(-1))

        return x * y.expand_as(x)


if __name__ == '__main__':
    import torchsummary
    in1 = torch.randn((4,64,12,12)).cuda()
    net = MECA(64).cuda()
    out = net(in1)
    print(out.size())
    # torchsummary.summary(net, input_size=(1,48,48), batch_size=-1, device='cuda')
    print('# discriminator parameters:', sum(param.numel() for param in net.parameters()))
    