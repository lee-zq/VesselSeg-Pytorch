import torch
import torch.nn.functional as F
import torch.nn as nn

drop = 0.25

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=True)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        if inplanes!= planes:
            self.conv0 = conv3x3(inplanes,planes)

        self.inplanes = inplanes
        self.planes = planes

        self.conv1 = conv3x3(planes, planes, stride)
        #self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        #self.conv2 = conv3x3(planes, planes)
        #self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.drop = nn.Dropout2d(p=drop)

    def forward(self, x):
        if self.inplanes != self.planes:
            x = self.conv0(x)
            x = F.relu(x)

        out = self.conv1(x)
        #out = self.bn1(out)
        out = self.relu(out)

        out = self.drop(out)

        out1 = self.conv1(out)
        #out1 = self.relu(out1)

        out2 = out1 + x

        return F.relu(out2)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Initial_LadderBlock(nn.Module):

    def __init__(self,planes,layers,kernel=3,block=BasicBlock,inplanes = 3):
        super().__init__()
        self.planes = planes
        self.layers = layers
        self.kernel = kernel

        self.padding = int((kernel-1)/2)
        self.inconv = nn.Conv2d(in_channels=inplanes,out_channels=planes,
                                kernel_size=3,stride=1,padding=1,bias=True)

        # create module list for down branch
        self.down_module_list = nn.ModuleList()
        for i in range(0,layers):
            self.down_module_list.append(block(planes*(2**i),planes*(2**i)))

        # use strided conv instead of pooling
        self.down_conv_list = nn.ModuleList()
        for i in range(0,layers):
            self.down_conv_list.append(nn.Conv2d(planes*2**i,planes*2**(i+1),stride=2,kernel_size=kernel,padding=self.padding))

        # create module for bottom block
        self.bottom = block(planes*(2**layers),planes*(2**layers))

        # create module list for up branch
        self.up_conv_list = nn.ModuleList()
        self.up_dense_list = nn.ModuleList()
        for i in range(0, layers):
            self.up_conv_list.append(nn.ConvTranspose2d(in_channels=planes*2**(layers-i), out_channels=planes*2**max(0,layers-i-1), kernel_size=3,
                                                        stride=2,padding=1,output_padding=1,bias=True))
            self.up_dense_list.append(block(planes*2**max(0,layers-i-1),planes*2**max(0,layers-i-1)))


    def forward(self, x):
        out = self.inconv(x)
        out = F.relu(out)

        down_out = []
        # down branch
        for i in range(0,self.layers):
            out = self.down_module_list[i](out)
            down_out.append(out)
            out = self.down_conv_list[i](out)
            out = F.relu(out)

        # bottom branch
        out = self.bottom(out)
        bottom = out

        # up branch
        up_out = []
        up_out.append(bottom)

        for j in range(0,self.layers):
            out = self.up_conv_list[j](out) + down_out[self.layers-j-1]
            #out = F.relu(out)
            out = self.up_dense_list[j](out)
            up_out.append(out)

        return up_out

class LadderBlock(nn.Module):

    def __init__(self,planes,layers,kernel=3,block=BasicBlock,inplanes = 3):
        super().__init__()
        self.planes = planes
        self.layers = layers
        self.kernel = kernel

        self.padding = int((kernel-1)/2)
        self.inconv = block(planes,planes)

        # create module list for down branch
        self.down_module_list = nn.ModuleList()
        for i in range(0,layers):
            self.down_module_list.append(block(planes*(2**i),planes*(2**i)))

        # use strided conv instead of poooling
        self.down_conv_list = nn.ModuleList()
        for i in range(0,layers):
            self.down_conv_list.append(nn.Conv2d(planes*2**i,planes*2**(i+1),stride=2,kernel_size=kernel,padding=self.padding))

        # create module for bottom block
        self.bottom = block(planes*(2**layers),planes*(2**layers))

        # create module list for up branch
        self.up_conv_list = nn.ModuleList()
        self.up_dense_list = nn.ModuleList()
        for i in range(0, layers):
            self.up_conv_list.append(nn.ConvTranspose2d(planes*2**(layers-i), planes*2**max(0,layers-i-1), kernel_size=3,
                                                        stride=2,padding=1,output_padding=1,bias=True))
            self.up_dense_list.append(block(planes*2**max(0,layers-i-1),planes*2**max(0,layers-i-1)))


    def forward(self, x):
        out = self.inconv(x[-1])

        down_out = []
        # down branch
        for i in range(0,self.layers):
            out = out + x[-i-1]
            out = self.down_module_list[i](out)
            down_out.append(out)

            out = self.down_conv_list[i](out)
            out = F.relu(out)

        # bottom branch
        out = self.bottom(out)
        bottom = out

        # up branch
        up_out = []
        up_out.append(bottom)

        for j in range(0,self.layers):
            out = self.up_conv_list[j](out) + down_out[self.layers-j-1]
            #out = F.relu(out)
            out = self.up_dense_list[j](out)
            up_out.append(out)

        return up_out

class Final_LadderBlock(nn.Module):

    def __init__(self,planes,layers,kernel=3,block=BasicBlock,inplanes = 3):
        super().__init__()
        self.block = LadderBlock(planes,layers,kernel=kernel,block=block)

    def forward(self, x):
        out = self.block(x)
        return out[-1]

class LadderNet(nn.Module):
    def __init__(self,inplanes=1,num_classes=2,layers=4,filters=10,):
        super().__init__()
        self.initial_block = Initial_LadderBlock(planes=filters,layers=layers,inplanes=inplanes)
        #self.middle_block = LadderBlock(planes=filters,layers=layers)
        self.final_block = Final_LadderBlock(planes=filters,layers=layers)
        self.final = nn.Conv2d(in_channels=filters,out_channels=num_classes,kernel_size=1)

    def forward(self,x):
        out = self.initial_block(x)
        #out = self.middle_block(out)
        out = self.final_block(out)
        out = self.final(out)
        #out = F.relu(out)
        out = F.softmax(out,dim=1)
        return out