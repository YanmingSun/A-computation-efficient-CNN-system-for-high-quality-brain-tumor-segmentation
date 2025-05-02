import torch
import torch.nn as nn
import torch.nn.functional as F


class Full_ReLU(nn.Module):
    def __init__(self, weight=1):
        super(Full_ReLU, self).__init__()
        self.weight = weight

    def forward(self, x):
        x1 = F.relu(x)
        x2 = F.relu(-x)
        x_out = torch.cat((x1, x2), 1)

        return x_out


class Net_Old_version(nn.Module):
    def __init__(self, num_modality=4, num_channels=16):
        super(Net_Old_version, self).__init__()
        self.full_relu = Full_ReLU()

        self.depthwise_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=num_modality, out_channels=int(num_channels/2), kernel_size=(3, 3), padding='same', padding_mode='reflect', groups=num_modality),
            nn.InstanceNorm2d(int(num_channels/2)))
        self.depthwise_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=(3, 3), padding='same',padding_mode='reflect', groups=num_channels),
            nn.InstanceNorm2d(num_channels))

        self.standard_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=num_modality, out_channels=int(num_channels / 2), kernel_size=(3, 3), padding='same', padding_mode='reflect'),
            nn.BatchNorm2d(int(num_channels / 2)))
        self.standard_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=int(num_channels / 2), kernel_size=(3, 3), padding='same', padding_mode='reflect'),
            nn.BatchNorm2d(int(num_channels / 2)))

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=num_channels * 3, out_channels=num_channels, kernel_size=(3, 3), padding='same', padding_mode='reflect'),
            nn.BatchNorm2d(num_channels))

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=num_channels * 2, out_channels=num_channels, kernel_size=(3, 3), padding='same',
                      padding_mode='reflect'),
            nn.BatchNorm2d(num_channels),
            nn.ReLU())
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=(3, 3), padding='same',
                      padding_mode='reflect'),
            nn.BatchNorm2d(num_channels),
            nn.ReLU())
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=(3, 3), padding='same',
                      padding_mode='reflect'),
            nn.BatchNorm2d(num_channels),
            nn.ReLU())

        self.conv7 = nn.Conv2d(in_channels=num_channels, out_channels=4, kernel_size=(3, 3), padding='same',
                      padding_mode='reflect')

        self.conv_1x1_1 = nn.Conv2d(in_channels=num_channels * 2, out_channels=num_channels, kernel_size=(1, 1), padding='same', padding_mode='reflect')
        self.conv_1x1_2 = nn.Conv2d(in_channels=num_channels * 3, out_channels=num_channels, kernel_size=(1, 1),
                                    padding='same', padding_mode='reflect')
        self.conv_1x1_3 = nn.Conv2d(in_channels=num_channels * 2, out_channels=num_channels, kernel_size=(1, 1),
                                    padding='same', padding_mode='reflect')

    def forward(self, x):
        x_dw_relu1 = self.full_relu(self.depthwise_conv1(x))
        x_dw_pool1 = F.max_pool2d(x_dw_relu1, kernel_size=[2, 2], stride=[2, 2])
        x_dw_relu2 = self.full_relu(self.depthwise_conv2(x_dw_pool1))
        x_dw_pool2 = F.max_pool2d(x_dw_relu2, kernel_size=[2, 2], stride=[2, 2])

        x_sd_relu1 = self.full_relu(self.standard_conv1(x))
        x_sd_pool1 = F.max_pool2d(x_sd_relu1, kernel_size=[2, 2], stride=[2, 2])
        x_sd_relu2 = self.full_relu(self.standard_conv2(x_sd_pool1))
        x_sd_pool2 = F.max_pool2d(x_sd_relu2, kernel_size=[2, 2], stride=[2, 2])

        x_relu1 = torch.cat((x_dw_relu1, x_sd_relu1), 1)
        x_relu2 = torch.cat((x_dw_relu2, x_sd_relu2), 1)
        x_pool2 = torch.cat((x_dw_pool2, x_sd_pool2), 1)

        x_relu3 = self.full_relu(self.conv3(x_pool2))
        x_pool3 = F.max_pool2d(x_relu3, kernel_size=[2, 2], stride=[2, 2])

        x_relu4 = self.conv4(x_pool3)
        x_upsample4 = F.interpolate(x_relu4, scale_factor=(2, 2), mode='bilinear', align_corners=True)
        x_bypass4 = self.conv_1x1_1(x_relu3)
        x_m4 = x_upsample4 + x_bypass4

        x_relu5 = self.conv5(x_m4)
        x_upsample5 = F.interpolate(x_relu5, scale_factor=(2, 2), mode='bilinear', align_corners=True)
        x_bypass5 = self.conv_1x1_2(x_relu2)
        x_m5 = x_upsample5 + x_bypass5

        x_relu6 = self.conv6(x_m5)
        x_upsample6 = F.interpolate(x_relu6, scale_factor=(2, 2), mode='bilinear', align_corners=True)
        x_bypass6 = self.conv_1x1_3(x_relu1)
        x_m6 = x_upsample6 + x_bypass6

        x_7 = self.conv7(x_m6)

        # x_out = F.softmax(x_7, dim=1)

        return x_7


