from typing import Callable, List, Optional

from torch import Tensor, nn

from .configuration import StunConfig, StunV2Config
from . import feature_extraction


__all__ = ['Stun']


class PixelShuffleConvBlock(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        upscale_factor: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        activation_layer: Optional[Callable[..., nn.Module]] = nn.ReLU,
        dilation: int = 1,
    ) -> None:
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        layers = [
            nn.PixelShuffle(upscale_factor),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
            ),
            activation_layer(),
        ]
        super(PixelShuffleConvBlock, self).__init__(*layers)


class RestoreBlock(nn.Sequential):
    expansion = 4

    def __init__(self, in_channels: int, stem_channels: int, head_channels: int) -> None:
        layers = [
            nn.Conv2d(in_channels, stem_channels, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(stem_channels, stem_channels, kernel_size=3, padding=1),
            nn.PReLU(),
        ]

        layers.append(
            PixelShuffleConvBlock(
                head_channels,
                head_channels * self.expansion,
                upscale_factor=4,
                activation_layer=nn.PReLU
            )
        )
        num_upsamples = 4
        for _ in range(1, num_upsamples):
            layers.append(
                PixelShuffleConvBlock(
                    head_channels,
                    head_channels * self.expansion,
                    upscale_factor=2,
                    activation_layer=nn.PReLU
                )
            )
        super(RestoreBlock, self).__init__(*layers)


class RestoreBlockV2(nn.Module):
    def __init__(
        self,
        channels: List[int],
        # neck_channels: int,
        selected_layers: List[int] = [0, 1, 2, 3],
    ) -> None:
        super(RestoreBlockV2, self).__init__()
        self.channels = [channels[i] for i in selected_layers]
        self.selected_layers = selected_layers
        # self.neck_channels = neck_channels
        self.num_layers = len(self.selected_layers)

        # self.residual_layers = nn.ModuleList([
        #     nn.Conv2d(
        #         in_channels,
        #         self.neck_channels,
        #         kernel_size=3,
        #     ) for in_channels in reversed(self.channels)
        # ])
        self.layers = []

        self.layers.append(
            ConvNormActivation(
                self.channels[-1],
                self.channels[-1],
                kernel_size=3,
                norm_layer=None,
                activation_layer=nn.PReLU,
            )
        )

        self.layers.extend([
            ConvPixelShuffleBlock(
                self.channels[i],
                self.channels[i+1],
                upscale_factor=2,
                activation_layer=nn.PReLU,
            ) for i in reversed(range(len(self.channels) - 1))
        ])

    def forward(self, x: List[Tensor]) -> Tensor:
        outputs = x[-1]

        residual = x[-1]
        outputs = self.layers[0](outputs)
        outputs = outputs + residual

        for i in range(1, self.num_layers):
            residual = x[-(i + 1)]
            outputs = self.layers[i](outputs)
            print(residual.shape, outputs.shape)
            outputs = outputs + residual

        return outputs


class ConvPixelShuffleBlock(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        upscale_factor: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        activation_layer: Optional[Callable[..., nn.Module]] = nn.ReLU,
        dilation: int = 1,
    ) -> None:
        assert upscale_factor % 2 == 0, 'upscale_factor are not even'
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        layers = [
            nn.Conv2d(
                in_channels,
                out_channels * (upscale_factor ** 2),
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
            ),
            nn.PixelShuffle(upscale_factor),
            activation_layer(),
        ]
        super(ConvPixelShuffleBlock, self).__init__(*layers)


class ConvPixelUnshuffleBlock(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        downscale_factor: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        activation_layer: Optional[Callable[..., nn.Module]] = nn.ReLU,
        dilation: int = 1,
    ) -> None:
        assert downscale_factor % 2 == 0, 'downscale_factor are not even'
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        layers = [
            nn.Conv2d(
                in_channels,
                int(out_channels * (downscale_factor ** -2)),
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
            ),
            nn.PixelUnshuffle(downscale_factor),
            activation_layer(),
        ]
        super(ConvPixelUnshuffleBlock, self).__init__(*layers)


class ConvNormActivation(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., nn.Module]] = nn.ReLU,
        dilation: int = 1,
    ) -> None:
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation=dilation,
                groups=groups,
                bias=norm_layer is None
            )
        ]
        if norm_layer is not None:
            layers.append(norm_layer(out_channels))
        if activation_layer is not None:
            layers.append(activation_layer())
        super().__init__(*layers)
        self.out_channels = out_channels


class TUNetX2(nn.Module):
    def __init__(self, in_channels: int, stem_channels: int, head_channels: int) -> None:
        super(TUNetX2, self).__init__()
        self.head_channels = head_channels

        self.restore_block = RestoreBlock(in_channels, stem_channels, head_channels)

        self.bi_x4 = nn.Upsample(scale_factor=4, mode='bicubic', align_corners=False)
        self.bi_x2 = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=False)

        self.ups_x2 = ConvPixelShuffleBlock(
            head_channels * self.restore_block.expansion,
            head_channels * 2,
            upscale_factor=2,
            kernel_size=3,
            activation_layer=nn.PReLU,
        )
        self.downs_x2 = ConvPixelUnshuffleBlock(
            head_channels * self.restore_block.expansion,
            head_channels * 2,
            downscale_factor=2,
            kernel_size=3,
            activation_layer=nn.PReLU,
        )

        self.conv2 = ConvNormActivation(
            head_channels * self.restore_block.expansion,
            head_channels * 2,
            kernel_size=3,
            norm_layer=nn.BatchNorm2d,
            activation_layer=nn.LeakyReLU
        )

        self.conv3 = ConvNormActivation(
            head_channels * 2,
            head_channels,
            kernel_size=1,
            norm_layer=None,
            activation_layer=nn.PReLU
        )
        self.conv4 = ConvNormActivation(
            head_channels,
            out_channels=1,
            kernel_size=3,
            norm_layer=None,
            activation_layer=nn.PReLU
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.restore_block(x)

        # Upsampling
        up1 = self.ups_x2(out)

        up2 = self.bi_x4(out)
        up2 = self.downs_x2(up2)

        up3 = self.bi_x2(out)
        up3 = self.conv2(up3)

        out = up1 + up2 + up3

        out = self.conv4(self.conv3(out))

        return out


class TUNetX4(nn.Module):
    def __init__(self, in_channels: int, stem_channels: int, head_channels: int) -> None:
        super(TUNetX4, self).__init__()
        self.head_channels = head_channels

        self.restore_block = RestoreBlock(in_channels, stem_channels, head_channels)

        self.bi_x4 = nn.Upsample(scale_factor=4, mode='bicubic', align_corners=False)
        self.bi_x2 = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=False)

        self.ups_x2_1 = ConvPixelShuffleBlock(
            head_channels * self.restore_block.expansion,
            head_channels * 2,
            upscale_factor=2,
            kernel_size=3,
            activation_layer=nn.PReLU,
        )
        self.ups_x2_2 = ConvPixelShuffleBlock(
            head_channels * self.restore_block.expansion,
            head_channels * 2,
            upscale_factor=2,
            kernel_size=3,
            activation_layer=nn.PReLU,
        )

        self.conv2 = ConvNormActivation(
            head_channels * self.restore_block.expansion,
            head_channels * 2,
            kernel_size=3,
            norm_layer=nn.BatchNorm2d,
            activation_layer=nn.LeakyReLU
        )

        self.conv3 = ConvNormActivation(
            head_channels * 2,
            head_channels,
            kernel_size=1,
            norm_layer=None,
            activation_layer=nn.PReLU
        )
        self.conv4 = ConvNormActivation(
            head_channels,
            out_channels=1,
            kernel_size=3,
            norm_layer=None,
            activation_layer=nn.PReLU
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.restore_block(x)

        # Upsampling
        up1 = self.bi_x2(out)
        up1 = self.ups_x2_2(up1)

        up2 = self.conv2(self.bi_x4(out))

        up3 = self.ups_x2_1(out)
        up3 = self.bi_x2(up3)

        out = up1 + up2 + up3

        out = self.conv4(self.conv3(out))

        return out


class TUNetX4V2(nn.Module):
    def __init__(self, channels: List[int], head_channels: int) -> None:
        super(TUNetX4V2, self).__init__()
        self.head_channels = head_channels

        self.restore = RestoreBlockV2(channels)
        in_channels = channels[0]

        self.bi_x4 = nn.Upsample(scale_factor=4, mode='bicubic', align_corners=False)
        self.bi_x2 = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=False)

        self.ups_x2_1 = ConvPixelShuffleBlock(
            # neck_channels,
            in_channels,
            head_channels * 2,
            upscale_factor=2,
            kernel_size=3,
            activation_layer=nn.PReLU,
        )
        self.ups_x2_2 = ConvPixelShuffleBlock(
            # neck_channels,
            in_channels,
            head_channels * 2,
            upscale_factor=2,
            kernel_size=3,
            activation_layer=nn.PReLU,
        )

        self.conv2 = ConvNormActivation(
            # neck_channels,
            in_channels,
            head_channels * 2,
            kernel_size=3,
            norm_layer=nn.BatchNorm2d,
            activation_layer=nn.LeakyReLU
        )

        self.conv3 = ConvNormActivation(
            head_channels * 2,
            head_channels,
            kernel_size=1,
            norm_layer=None,
            activation_layer=nn.PReLU
        )
        self.conv4 = ConvNormActivation(
            head_channels,
            out_channels=1,
            kernel_size=3,
            norm_layer=None,
            activation_layer=nn.PReLU
        )

    def forward(self, x: List[Tensor]) -> Tensor:
        x = self.restore(x)

        # Upsampling
        up1 = self.bi_x2(x)
        up1 = self.ups_x2_2(up1)

        up2 = self.conv2(self.bi_x4(x))

        up3 = self.ups_x2_1(x)
        up3 = self.bi_x2(up3)

        out = up1 + up2 + up3

        out = self.conv4(self.conv3(out))

        return out


class Stun(nn.Module):
    def __init__(self, cfg: StunConfig, **kwargs) -> None:
        super(Stun, self).__init__()

        self.cfg = cfg
        self.in_channels = cfg.in_channels

        self.task = cfg.task
        assert self.task == 'x2' or self.task == 'x4', 'wrong task'

        if self.task == 'x2':
            self.head = TUNetX2(self.in_channels, cfg.stem_channels, cfg.head_channels)
        elif self.task == 'x4':
            self.head = TUNetX4(self.in_channels, cfg.stem_channels, cfg.head_channels)

        self.backbone = feature_extraction.__dict__[cfg.backbone_name](**kwargs)

    def forward(self, x: Tensor) -> Tensor:
        x = self.backbone(x)
        x = self.head(x)
        return x


class StunV2(nn.Module):
    def __init__(self, cfg: StunV2Config, **kwargs) -> None:
        super(StunV2, self).__init__()

        self.cfg = cfg

        self.backbone = feature_extraction.__dict__[cfg.backbone_name](**kwargs)
        backbone_channels = self.backbone.channels

        self.task = cfg.task
        assert self.task == 'x2' or self.task == 'x4', 'wrong task'

        if self.task == 'x2':
            raise NotImplementedError
        elif self.task == 'x4':
            self.head = TUNetX4V2(backbone_channels, cfg.head_channels)

        # self.neck = RestoreBlockV2(backbone_channels, cfg.neck_channels)

    def forward(self, x: Tensor) -> Tensor:
        x = self.backbone(x)
        # x = self.neck(x)
        x = self.head(x)
        return x
