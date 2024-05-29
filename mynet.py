# coding=gbk
# coding=gbk
import torch
import torch.nn as nn
import torch.nn.functional as F

# EfficientNet中残差块设计，称之为MBConvBlock
class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio, kernel_size, stride, se_ratio=0.25):
        super(MBConvBlock, self).__init__()

        self.expand_ratio = max(1, expand_ratio)  # Ensure expand_ratio is at least 1
        self.kernel_size = kernel_size
        self.stride = stride
        self.se_ratio = se_ratio
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_res_connect = self.stride == 1 and self.in_channels == self.out_channels

        # Expansion phase
        expanded_channels = int(in_channels * self.expand_ratio)  # Dynamically calculate expanded_channels
        self.expand_conv = nn.Conv2d(in_channels, expanded_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expanded_channels)

        # Depth-wise convolution phase
        self.depthwise_conv = nn.Conv2d(expanded_channels, expanded_channels, kernel_size=kernel_size,
                                        stride=stride, padding=kernel_size // 2, groups=expanded_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(expanded_channels)

        # Squeeze and Excitation phase
        if self.se_ratio:
            se_channels = int(expanded_channels * self.se_ratio)  # Dynamically calculate se_channels
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(expanded_channels, se_channels, kernel_size=1),
                nn.ReLU(inplace=False),
                nn.Conv2d(se_channels, expanded_channels, kernel_size=1),
                nn.Sigmoid()
            )

        # Output phase
        self.project_conv = nn.Conv2d(expanded_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x

        # Expansion and depth-wise convolution
        x = self.expand_conv(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.depthwise_conv(x)
        x = self.bn2(x)
        x = F.relu(x)

        # Squeeze and Excitation
        if self.se_ratio:
            se_weights = self.se(x)
            x = x * se_weights

        # Output phase
        x = self.project_conv(x)
        x = self.bn3(x)

        # Skip connection if possible
        if self.use_res_connect:
            x += identity

        return x

class RegBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super(RegBlock, self).__init__()
        mid_channels = out_channels // 2

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, groups=groups,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)

        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )

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

        residual = self.downsample(residual)

        out += residual
        out = self.relu(out)

        return out


class ressblock(nn.Module):
    def __init__(self, in_channels, channels, stride=1, dilation=1):
        super(ressblock, self).__init__()
        self.stride = stride
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 使用全局平均池化来计算注意力权重
        self.fc = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True)
        self.bn_fc = nn.BatchNorm2d(channels)
        self.fc_out = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True)
        self.sigmoid = nn.Sigmoid()

        # 添加一个匹配残差的1x1卷积层
        self.residual_conv = nn.Conv2d(in_channels, channels, kernel_size=1, stride=stride, padding=0, bias=False)
        self.bn_residual = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        # 计算注意力权重
        attn = self.avg_pool(x)
        attn = self.fc(attn)
        attn = self.bn_fc(attn)
        attn = self.relu(attn)
        attn = self.fc_out(attn)
        attn = self.sigmoid(attn)  # 使用sigmoid激活函数

        # 将注意力权重应用到特征图上
        x = x * attn

        # 调整残差的形状以便相加
        if residual.shape[1] != x.shape[1] or self.stride != 1:
            residual = self.residual_conv(residual)
            residual = self.bn_residual(residual)
        # 将残差连接
        x += residual
        x = self.relu(x)

        return x


class KenBlock(nn.Module):
    def __init__(self, in_channels, channels, stride=1, dilation=1):
        super(KenBlock, self).__init__()
        self.stride = stride  # 添加stride作为类的属性
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.residual_attention = ressblock(channels, channels, stride=stride, dilation=dilation)
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

        # 添加一个匹配残差的1x1卷积层
        self.residual_conv = nn.Conv2d(in_channels, channels, kernel_size=1, stride=stride, padding=0, bias=False)
        self.bn_residual = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.residual_attention(x)

        x = self.conv3(x)
        x = self.bn3(x)

        # 调整残差的形状以便相加
        if residual.shape[1] != x.shape[1] or self.stride != 1:
            residual = self.residual_conv(residual)
            residual = self.bn_residual(residual)

        # 将残差连接
        x += residual
        x = self.relu(x)

        return x


class HakiandRegNet_8(nn.Module):
    def __init__(self):
        super(HakiandRegNet_8, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = nn.Sequential(
            KenBlock(64, 64, 1),
            RegBlock(64, 64, 1, 2)
        )

        self.layer2 = nn.Sequential(
            KenBlock(64, 128, 2),
            RegBlock(128, 128, 1, 2)
        )

        self.layer3 = nn.Sequential(
            KenBlock(128, 256, 2),
            RegBlock(256, 256, 1, 2)
        )

        self.layer4 = nn.Sequential(
            KenBlock(256, 512, 2),
            RegBlock(512, 512, 1, 2)
        )

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(512, 20)  # 输出20分类

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.reshape(x.shape[0], -1)
        out = self.fc(out)
        return out

class HakiNet_8(nn.Module):
    def __init__(self):
        super(HakiNet_8, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = nn.Sequential(
            KenBlock(64, 64, 1),
            KenBlock(64, 64, 1)

        )

        self.layer2 = nn.Sequential(
            KenBlock(64, 128, 2),
            KenBlock(128, 128, 1)

        )

        self.layer3 = nn.Sequential(
            KenBlock(128, 256, 2),
            KenBlock(256, 256, 1)

        )

        self.layer4 = nn.Sequential(
            KenBlock(256, 512, 2),
            KenBlock(512, 512, 1)

        )

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(512, 20)  # 输出20分类

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.reshape(x.shape[0], -1)
        out = self.fc(out)
        return out

class HakiNet_4(nn.Module):
    def __init__(self):
        super(HakiNet_4, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = nn.Sequential(
            KenBlock(64, 64, 1)
        )

        self.layer2 = nn.Sequential(
            KenBlock(64, 128, 2)
        )

        self.layer3 = nn.Sequential(
            KenBlock(128, 256, 2)
        )

        self.layer4 = nn.Sequential(
            KenBlock(256, 512, 2)
        )

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(512, 20)  # 输出20分类

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.reshape(x.shape[0], -1)
        out = self.fc(out)
        return out

class RegNet_8(nn.Module):
    def __init__(self):
        super(RegNet_8, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = nn.Sequential(
            RegBlock(64, 64, 1, 8),
            RegBlock(64, 64, 1, 8)
        )

        self.layer2 = nn.Sequential(
            RegBlock(64, 128, 2, 8),
            RegBlock(128, 128, 1, 8)
        )

        self.layer3 = nn.Sequential(
            RegBlock(128, 256, 2, 8),
            RegBlock(256, 256, 1, 8)
        )

        self.layer4 = nn.Sequential(
            RegBlock(256, 512, 2, 8),
            RegBlock(512, 512, 1, 8)
        )

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(512, 20)  # 输出20分类

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.reshape(x.shape[0], -1)
        out = self.fc(out)
        return out


class MiNet_8(nn.Module):
    def __init__(self):
        super(MiNet_8, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 使用MBConvBlock替换KenBlock
        self.layer1 = nn.Sequential(
            MBConvBlock(64, 64, expand_ratio=1, kernel_size=3, stride=1),
            MBConvBlock(64, 64, expand_ratio=1, kernel_size=3, stride=1)
        )

        self.layer2 = nn.Sequential(
            MBConvBlock(64, 128, expand_ratio=6, kernel_size=3, stride=2),
            MBConvBlock(128, 128, expand_ratio=6, kernel_size=3, stride=1)
        )

        self.layer3 = nn.Sequential(
            MBConvBlock(128, 256, expand_ratio=6, kernel_size=3, stride=2),
            MBConvBlock(256, 256, expand_ratio=6, kernel_size=3, stride=1)
        )

        self.layer4 = nn.Sequential(
            MBConvBlock(256, 512, expand_ratio=6, kernel_size=3, stride=2),
            MBConvBlock(512, 512, expand_ratio=6, kernel_size=3, stride=1)
        )

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(512, 20)  # 输出20分类

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


class VGG16(nn.Module):
    def __init__(self, vgg):
        super(VGG16, self).__init__()
        self.features = self._make_layers(vgg)
        self.dense = nn.Sequential(
            nn.Linear(1024, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
        )
        self.classifier = nn.Linear(4096, 20)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.dense(out)
        out = self.classifier(out)
        return out

    def _make_layers(self, vgg):
        layers = []
        in_channels = 3
        for x in vgg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2,
                                        stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x,
                                     kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x

        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)



