"""
ResNet18 手写实现 —— 基于 PaddlePaddle
不调用框架预封装模型，完全从 BasicBlock 组装
"""

import paddle
import paddle.nn as nn


class BasicBlock(nn.Layer):
    """ResNet18 基础残差块：两个 3×3 卷积 + 跳跃连接"""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2D(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias_attr=False)
        self.bn1 = nn.BatchNorm2D(out_channels)
        self.conv2 = nn.Conv2D(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias_attr=False)
        self.bn2 = nn.BatchNorm2D(out_channels)
        self.relu = nn.ReLU()

        # 当输入/输出尺寸或通道数不匹配时，用 1×1 卷积对齐
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2D(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias_attr=False),
                nn.BatchNorm2D(out_channels)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + identity)
        return out


class ResNet18(nn.Layer):
    """手写 ResNet18 — 自适应全局平均池化保证维度兼容"""

    def __init__(self, num_classes=6):
        super().__init__()
        self.in_channels = 64

        # 初始卷积层
        self.conv1 = nn.Conv2D(3, 64, kernel_size=7, stride=2, padding=3,
                               bias_attr=False)
        self.bn1 = nn.BatchNorm2D(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)

        # 四个残差层
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        # 自适应全局平均池化 → 不依赖固定输入尺寸
        self.avgpool = nn.AdaptiveAvgPool2D((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, out_channels, blocks, stride):
        layers = []
        # 第一个 block 可能带 stride 和通道变化
        layers.append(BasicBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        # 后续 block 保持尺寸不变
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = paddle.flatten(x, start_axis=1)
        x = self.fc(x)
        return x


def create_model(num_classes: int) -> ResNet18:
    """工厂函数：创建 ResNet18 实例"""
    return ResNet18(num_classes=num_classes)
