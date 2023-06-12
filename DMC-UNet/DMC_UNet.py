from torch import nn
import torch
import torch.nn.functional as F
#=================================


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 2) / 4


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h1 = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_h2 = nn.AdaptiveMaxPool2d((None, 1))
        self.pool_w1 = nn.AdaptiveAvgPool2d((1, None))
        self.pool_w2 = nn.AdaptiveMaxPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

        self.fc1 = nn.Conv2d(inp, inp // reduction, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(inp // reduction, inp, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h1 = self.pool_h1(x)
        x_h2 = self.pool_h2(x)
        avg_out = self.fc2(self.relu(self.fc1(x_h1)))
        max_out = self.fc2(self.relu(self.fc1(x_h2)))
        x_A = self.sigmoid(avg_out + max_out)

        x_w1 = self.pool_w1(x)
        x_w2 = self.pool_w2(x)
        avg_out2 = self.fc2(self.relu(self.fc1(x_w1)))
        max_out2 = self.fc2(self.relu(self.fc1(x_w2)))
        y_A = self.sigmoid(avg_out2 + max_out2).permute(0, 1, 3, 2)

        y = torch.cat([x_A, y_A], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_A, y_A = torch.split(y, [h, w], dim=2)
        y_A = y_A.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_A).sigmoid()
        a_w = self.conv_w(y_A).sigmoid()

        out = identity * a_w * a_h
        return out

#深度可分离卷积
class DepthWiseConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DepthWiseConv, self).__init__()
        self.depth_point_conv = nn.Sequential(
            # 逐通道卷积
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, stride=1, padding=1, groups=in_channel),# groups是一个数，当groups=in_channel时,表示做逐通道卷积
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
            # 逐点卷积
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1, padding=0, groups=1)
        )
    def forward(self, input):
        out = self.depth_point_conv(input)
        return out


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            DepthWiseConv(in_ch, out_ch),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            DepthWiseConv(out_ch, out_ch),
            nn.BatchNorm2d(out_ch)
        )
        self.resconv = nn.Sequential(
            DepthWiseConv(in_ch, out_ch),
            nn.BatchNorm2d(out_ch)
        )
        self.action = nn.ReLU(inplace=True)
    def forward(self, input):
        x1 = self.conv(input)
        x2 = self.resconv(input)
        x3 = x1 + x2
        x3 = self.action(x3)
        return x3

class ASPP(nn.Module):
    def __init__(self, in_channel):
        depth = in_channel
        super(ASPP, self).__init__()
        self.conv = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block1 = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block6 = nn.Conv2d(in_channel, depth, 3, 1, padding=2, dilation=2)
        self.atrous_block12 = nn.Conv2d(in_channel, depth, 3, 1, padding=4, dilation=4)
        self.atrous_block18 = nn.Conv2d(in_channel, depth, 3, 1, padding=8, dilation=8)
        self.conv_1x1_output = nn.Conv2d(depth * 4, depth*2, 1, 1)
        self.batchnorm = nn.BatchNorm2d(in_channel*2)
        self.action = nn.ReLU(inplace=True)
    def forward(self, x):
        atrous_block1 = self.atrous_block1(x)
        atrous_block6 = self.atrous_block6(x)
        atrous_block12 = self.atrous_block12(x)
        atrous_block18 = self.atrous_block18(x)
        cat = torch.cat([atrous_block1, atrous_block6,atrous_block12, atrous_block18], dim=1)
        net = self.conv_1x1_output(cat)
        net = self.batchnorm(net)
        net = self.action(net)
        return net


class TransConv(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(TransConv, self).__init__()
        self.aspp = ASPP(in_ch)
        self.up = nn.PixelShuffle(2)
    def forward(self,input):
        x1 = self.aspp(input)
        x2 = self.up(x1)
        return x2

class resunet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(resunet, self).__init__()
        self.conv1 = DoubleConv(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2, stride=2)
        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2, stride=2)
        self.conv5 = DoubleConv(512, 1024)
        self.up6 = nn.ConvTranspose2d(1024, 512, 2,2)
        #self.up6 = TransConv(1024, 512)
        self.conv6 = DoubleConv(1024, 512)
        self.up7 = nn.ConvTranspose2d(512, 256, 2,2)
        #self.up7 = TransConv(512, 256)
        self.conv7 = DoubleConv(512, 256)
        self.up8 = nn.ConvTranspose2d(256, 128, 2,2)
        #self.up8 = TransConv(256, 128)
        self.conv8 = DoubleConv(256, 128)
        self.up9 = nn.ConvTranspose2d(128, 64, 2,2)
        #self.up9 = TransConv(128, 64)
        self.conv9 = DoubleConv(128, 64)
        self.conv10 = nn.Conv2d(64, out_ch, 1)
        self.dropout = nn.Dropout2d(p=0.5)
        self.up6_1 = TransConv(1024, 512)
        self.up7_1 = TransConv(512, 256)
        self.up8_1 = TransConv(256, 128)
        self.up9_1 = TransConv(128, 64)
        self.conv6_1 = DoubleConv(512,512)
        self.conv7_1 = DoubleConv(256,256)
        self.conv8_1 = DoubleConv(128,128)
        self.conv9_1 = DoubleConv(64,64)
        self.CDA1 = CoordAtt(512,512)
        self.CDA2 = CoordAtt(256, 256)
        self.CDA3 = CoordAtt(128, 128)
        # self.CDA4 = CoordAtt(64, 64)
        # self.CDA5 = CoordAtt(1024, 1024)

    def weight_init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
        # 也可以判断是否为conv2d，使用相应的初始化方式
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        # 是否为批归一化层
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        mid1 = self.dropout(c4)
        p4 = self.pool4(mid1)
        c5 = self.conv5(p4)
        mid2 = self.dropout(c5)
        up_6 = self.up6(mid2)
        up_6_1 = self.up6_1(mid2)
        merge6 = torch.cat([up_6, c4], dim=1)
        c6 = self.conv6(merge6)
        c6_aspp = self.conv6_1(up_6_1)
        c6_1 = c6 * c6_aspp
        cda1 = self.CDA1(c6_1)
        up_7 = self.up7(cda1)
        up_7_1 = self.up7_1(cda1)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(merge7)
        c7_aspp = self.conv7_1(up_7_1)
        c7_1 = c7 * c7_aspp
        cda2 = self.CDA2(c7_1)
        up_8 = self.up8(cda2)
        up_8_1 = self.up8_1(cda2)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)
        c8_aspp = self.conv8_1(up_8_1)
        c8_1 = c8 * c8_aspp
        cda3 = self.CDA3(c8_1)
        up_9 = self.up9(cda3)
        up_9_1 = self.up9_1(cda3)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)
        c9_aspp = self.conv9_1(up_9_1)
        c9_1 = c9 * c9_aspp
        c10 = self.conv10(c9_1)
        return c10

#==================================================================================
if __name__ == '__main__':
    model = resunet(1, 1)
    x = torch.randn([8,1,128,128])
    out = model(x)
    print(out.shape)
