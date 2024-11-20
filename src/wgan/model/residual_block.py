# Author: Joshua Park
# Modified by: Akira Kudo
# Created: 2024/11/01
# Last Updated: 2024/11/16

"""
Implementation of residual blocks for discriminator and generator.
We follow the official SNGAN Chainer implementation as closely as possible:
https://github.com/pfnet-research/sngan_projection
"""

import torch.nn as nn
import torch.nn.functional as F


class GBlock(nn.Module):
    r"""
    Residual block for generator.

    Uses linear (rather than nearest) interpolation, and align_corners
    set to False. This is as per how torchvision does upsampling, as seen in:
    https://github.com/pytorch/vision/blob/master/torchvision/models/segmentation/_utils.py

    Attributes:
        in_channels (int): The channel size of input feature map.
        out_channels (int): The channel size of output feature map.
        hidden_channels (int): The channel size of intermediate feature maps.
        upsample (bool): If True, upsamples the input feature map.
        num_classes (int): If more than 0, uses conditional batch norm instead.
        spectral_norm (bool): If True, uses spectral norm for convolutional layers.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels=None,
                 upsample=False,
                 num_classes=0,
                 spectral_norm=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels if hidden_channels is not None else out_channels
        self.learnable_sc = in_channels != out_channels or upsample
        self.upsample = upsample
        self.padding = nn.ReflectionPad1d((4,4))

        self.num_classes = num_classes
        self.spectral_norm = spectral_norm

        # Build the layers
        # Note: Can't use something like self.conv = SNConv1d to save code length
        # this results in somehow spectral norm working worse consistently.
        self.c1 = nn.Conv1d(self.in_channels,
                            self.hidden_channels,
                            3,
                            1)
        self.c2 = nn.Conv1d(self.hidden_channels,
                            self.out_channels,
                            3,
                            1)

        if self.num_classes == 0:
            self.b1 = nn.BatchNorm1d(self.in_channels)
            self.b2 = nn.BatchNorm1d(self.hidden_channels)

        self.activation = nn.LeakyReLU(0.2)

        nn.init.normal_(self.c1.weight.data, 0.0, 0.02)
        nn.init.normal_(self.c2.weight.data, 0.0, 0.02)

        # Shortcut layer
        if self.learnable_sc:
            self.c_sc = nn.Conv1d(in_channels,
                                  out_channels,
                                  1,
                                  1,
                                  padding=0)

            nn.init.normal_(self.c_sc.weight.data, 0.0, 0.02)

    def _upsample_conv(self, x, conv):
        r"""
        Helper function for performing convolution after upsampling.
        """
        return conv(
            F.interpolate(x,
                          scale_factor=2,
                          mode='linear',
                          align_corners=False))

    def _residual(self, x):
        r"""
        Helper function for feedforwarding through main layers.
        """
        h = x
        h = self._upsample_conv(h, self.c1) if self.upsample else self.c1(h)
        h = self.b1(h)
        h = self.activation(h)
#         h = self.c2(h)
#         h = self.b2(h)
#         h = self.activation(h)

        return h

    # def _residual_conditional(self, x, y):
    #     r"""
    #     Helper function for feedforwarding through main layers, including conditional BN.
    #     """
    #     h = x
    #     h = self.b1(h, y)
    #     h = self.activation(h)
    #     h = self._upsample_conv(h, self.c1) if self.upsample else self.c1(h)
    #     h = self.b2(h, y)
    #     h = self.activation(h)
    #     h = self.c2(h)

    #     return h

    def _shortcut(self, x):
        r"""
        Helper function for feedforwarding through shortcut layers.
        """
        if self.learnable_sc:
            x = self._upsample_conv(
                x, self.c_sc) if self.upsample else self.c_sc(x)
            return x
        else:
            return x

    def forward(self, x, y=None):
        r"""
        Residual block feedforward function.
        """
        if y is None:
            return self._residual(x)
        #+ self._shortcut(x)

        else:
            return self._residual_conditional(x, y)
        #+ self._shortcut(x)


class DBlock(nn.Module):
    """
    Residual block for discriminator.

    Attributes:
        in_channels (int): The channel size of input feature map.
        out_channels (int): The channel size of output feature map.
        hidden_channels (int): The channel size of intermediate feature maps.
        downsample (bool): If True, downsamples the input feature map.
        spectral_norm (bool): If True, uses spectral norm for convolutional layers.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 hidden_channels=None,
                 downsample=False,
                 spectral_norm=True,
                 reflectionPad = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels if hidden_channels is not None else in_channels
        self.downsample = downsample
        self.learnable_sc = (in_channels != out_channels) or downsample
        self.spectral_norm = spectral_norm
        self.kernel_size = kernel_size
        self.padding = nn.ReflectionPad1d((padding,padding))
        self.reflectionPad = reflectionPad

        # Build the layers
        self.c1 = nn.Conv1d(self.in_channels, self.out_channels, self.kernel_size, stride)
        self.c2 = nn.Conv1d(self.hidden_channels, self.out_channels, self.kernel_size, stride)

        self.activation = nn.LeakyReLU(0.2)

        nn.init.normal_(self.c1.weight.data, 0.0, 0.02)
        nn.init.normal_(self.c2.weight.data, 0.0, 0.02)

        # Shortcut layer
        if self.learnable_sc:
            self.c_sc = nn.Conv1d(in_channels, out_channels, 1, 1, 0)

            nn.init.normal_(self.c_sc.weight.data, 0.0, 0.02)

    def _residual(self, x):
        """
        Helper function for feedforwarding through main layers.
        """
        h = x
        h = self.c1(h)
        h = self.activation(h)
        if self.downsample:
            h = F.avg_pool1d(h, 2)
#         h = self.c2(h)
#         h = self.activation(h)
        
        return h

    def _shortcut(self, x):
        """
        Helper function for feedforwarding through shortcut layers.
        """
        if self.learnable_sc:
            x = self.c_sc(x)
            return F.avg_pool1d(x, 2) if self.downsample else x

        else:
            return x

    def forward(self, x):
        """
        Residual block feedforward function.
        """
        return self._residual(x)
    #+ self._shortcut(x)


class DBlockOptimized(nn.Module):
    """
    Optimized residual block for discriminator. This is used as the first residual block,
    where there is a definite downsampling involved. Follows the official SNGAN reference implementation
    in chainer.

    Attributes:
        in_channels (int): The channel size of input feature map.
        out_channels (int): The channel size of output feature map.
        spectral_norm (bool): If True, uses spectral norm for convolutional layers.
    """
    def __init__(self, in_channels, out_channels, spectral_norm=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spectral_norm = spectral_norm

        # Build the layers
        self.c1 = nn.Conv1d(self.in_channels, self.out_channels, 1, 1, 0)
        self.c2 = nn.Conv1d(self.out_channels, self.out_channels, 1, 1, 0)
        self.c_sc = nn.Conv1d(self.in_channels, self.out_channels, 1, 1, 0)

        self.activation = nn.LeakyReLU(0.2)

        nn.init.normal_(self.c1.weight.data, 0.0, 0.02)
        nn.init.normal_(self.c2.weight.data, 0.0, 0.02)
        nn.init.normal_(self.c_sc.weight.data, 0.0, 0.02)

    def _residual(self, x):
        """
        Helper function for feedforwarding through main layers.
        """
        h = x
        h = self.c1(h)
        h = self.activation(h)
        h = self.c2(h)
        h = F.avg_pool1d(h, 2)
        return h

    def _shortcut(self, x):
        """
        Helper function for feedforwarding through shortcut layers.
        """
        return self.c_sc(F.avg_pool1d(x, 2))

    def forward(self, x):
        """
        Residual block feedforward function.
        """
        return self._residual(x)
    #+ self._shortcut(x)

"""
ResBlocks for WGAN-GP.
"""

class GBlock(GBlock):
    r"""
    Residual block for generator.
    Modifies original resblock definitions with small changes.

    Uses linear (rather than nearest) interpolation, and align_corners
    set to False. This is as per how torchvision does upsampling, as seen in:
    https://github.com/pytorch/vision/blob/master/torchvision/models/segmentation/_utils.py

    Attributes:
        in_channels (int): The channel size of input feature map.
        out_channels (int): The channel size of output feature map.
        hidden_channels (int): The channel size of intermediate feature maps.
        upsample (bool): If True, upsamples the input feature map.
        num_classes (int): If more than 0, uses conditional batch norm instead.
        spectral_norm (bool): If True, uses spectral norm for convolutional layers.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels=None,
                 upsample=False,
                 num_classes=0,
                 spectral_norm=False,
                 **kwargs):
        super().__init__(in_channels=in_channels,
                         out_channels=out_channels,
                         hidden_channels=hidden_channels,
                         upsample=upsample,
                         num_classes=num_classes,
                         spectral_norm=spectral_norm,
                         **kwargs)

        # Redefine shortcut layer without act.
        if self.learnable_sc:
            self.c_sc = nn.Conv1d(self.in_channels,
                                  self.out_channels,
                                  1,
                                  1,
                                  padding=0)


class DBlock(DBlock):
    r"""
    Residual block for discriminator.

    Modifies original resblock definition by including layer norm and removing
    act for shortcut. Convs are LN-ReLU-Conv. See official TF code:
    https://github.com/igul222/improved_wgan_training/blob/master/gan_cifar_resnet.py#L105

    Attributes:
        in_channels (int): The channel size of input feature map.
        out_channels (int): The channel size of output feature map.
        hidden_channels (int): The channel size of intermediate feature maps.
        downsample (bool): If True, downsamples the input feature map.
        spectral_norm (bool): If True, uses spectral norm for convolutional layers.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 hidden_channels=None,
                 downsample=False,
                 spectral_norm=False,
                 reflectionPad=False,
                 **kwargs):
        super().__init__(in_channels=in_channels,
                         out_channels=out_channels,
                         kernel_size=kernel_size,
                         stride=stride,
                         padding=padding,
                         hidden_channels=hidden_channels,
                         downsample=downsample,
                         spectral_norm=spectral_norm,
                         reflectionPad=reflectionPad,
                         **kwargs)

        # Redefine shortcut layer without act.
        # TODO: Maybe can encapsulate defining of learnable sc in a fn
        # then override it later? Might be cleaner.
        if self.learnable_sc:
            self.c_sc = nn.Conv1d(self.in_channels, self.out_channels, 1, 1, 0)

        self.norm1 = None
        self.norm2 = None

    # TODO: Verify again. Interestingly, LN has no effect on FID. Not using LN
    # has almost no difference in FID score.
    # def residual(self, x):
    #     r"""
    #     Helper function for feedforwarding through main layers.
    #     """
    #     if self.norm1 is None:
    #         self.norm1 = nn.LayerNorm(
    #             [self.in_channels, x.shape[2], x.shape[3]])

    #     h = x
    #     h = self.norm1(h)
    #     h = self.activation(h)
    #     h = self.c1(h)

    #     if self.norm2 is None:
    #         self.norm2 = nn.LayerNorm(
    #             [self.hidden_channels, h.shape[2], h.shape[3]])

    #     h = self.norm2(h)
    #     h = self.activation(h)
    #     h = self.c2(h)
    #     if self.downsample:
    #         h = F.avg_pool2d(h, 2)

    #     return h


class DBlockOptimized(DBlockOptimized):
    r"""
    Optimized residual block for discriminator.

    Does not have any normalisation. See official TF Code:
    https://github.com/igul222/improved_wgan_training/blob/master/gan_cifar_resnet.py#L139

    Attributes:
        in_channels (int): The channel size of input feature map.
        out_channels (int): The channel size of output feature map.
        spectral_norm (bool): If True, uses spectral norm for convolutional layers.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 spectral_norm=False,
                 **kwargs):
        super().__init__(in_channels=in_channels,
                         out_channels=out_channels,
                         spectral_norm=spectral_norm,
                         **kwargs)

        # Redefine shortcut layer
        self.c_sc = nn.Conv1d(self.in_channels, self.out_channels, 1, 1, 0)