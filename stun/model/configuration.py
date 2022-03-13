__all__ = [
    'StunConfig', 'stun_x4_t_config', 'stun_x4_t_pad_config', 'stun_x4_s_config', 'stun_x2_t_config',
    'stun_x2_t_1k_config', 'stun_x2_s_config', 'stun_x2_s_1k_config', 'stun_x2_b_config',
    'stun_x2_b_1k_config', 'stun_x2_b_22k_to_1k_config', 'stun_x2_b_22k_config', 'stun_x2_l_1k_config',
]


class StunConfig:
    def __init__(
        self,
        task: str,
        backbone_name: str,
        in_channels: int,
        stem_channels: int,
        head_channels: int,
        **kwargs
    ) -> None:
        self.task = task
        self.in_channels = in_channels
        self.backbone_name = backbone_name
        self.stem_channels = stem_channels
        self.head_channels = head_channels


def stun_x4_t_config(**kwargs) -> StunConfig:
    return StunConfig(
        task='x4',
        backbone_name='swin_transformer_96_t',
        in_channels=768,
        stem_channels=1024,
        head_channels=64,
        **kwargs
    )


def stun_x4_t_pad_config(**kwargs) -> StunConfig:
    return StunConfig(
        task='x4',
        backbone_name='swin_transformer_160_t',
        in_channels=768,
        stem_channels=1024,
        head_channels=64,
        **kwargs
    )


def stun_x4_s_config(**kwargs) -> StunConfig:
    return StunConfig(
        task='x4',
        backbone_name='swin_transformer_96_s',
        in_channels=768,
        stem_channels=1024,
        head_channels=64,
        **kwargs
    )


def stun_x2_t_config(**kwargs) -> StunConfig:
    return StunConfig(
        task='x2',
        backbone_name='swin_transformer_224_t',
        pretrained=False,
        in_channels=768,
        stem_channels=256,
        head_channels=16,
        **kwargs
    )


def stun_x2_t_1k_config(**kwargs) -> StunConfig:
    return StunConfig(
        task='x2',
        backbone_name='swin_transformer_224_t',
        pretrained=True,
        in_channels=768,
        stem_channels=256,
        head_channels=16,
        **kwargs
    )


def stun_x2_s_config(**kwargs) -> StunConfig:
    return StunConfig(
        task='x2',
        backbone_name='swin_transformer_224_s',
        pretrained=False,
        in_channels=768,
        stem_channels=256,
        head_channels=16,
        **kwargs
    )


def stun_x2_s_1k_config(**kwargs) -> StunConfig:
    return StunConfig(
        task='x2',
        backbone_name='swin_transformer_224_s',
        pretrained=True,
        in_channels=768,
        stem_channels=256,
        head_channels=16,
        **kwargs
    )


def stun_x2_b_config(**kwargs) -> StunConfig:
    return StunConfig(
        task='x2',
        backbone_name='swin_transformer_224_b_1k',
        pretrained=False,
        in_channels=1024,
        stem_channels=256,
        head_channels=16,
        **kwargs
    )


def stun_x2_b_1k_config(**kwargs) -> StunConfig:
    return StunConfig(
        task='x2',
        backbone_name='swin_transformer_224_b_1k',
        pretrained=True,
        in_channels=1024,
        stem_channels=256,
        head_channels=16,
        **kwargs
    )


def stun_x2_b_22k_to_1k_config(**kwargs) -> StunConfig:
    return StunConfig(
        task='x2',
        backbone_name='swin_transformer_224_b_22k_to_1k',
        pretrained=True,
        in_channels=1024,
        stem_channels=256,
        head_channels=16,
        **kwargs
    )


def stun_x2_b_22k_config(**kwargs) -> StunConfig:
    return StunConfig(
        task='x2',
        backbone_name='swin_transformer_224_b_22k',
        pretrained=True,
        in_channels=1024,
        stem_channels=256,
        head_channels=16,
        **kwargs
    )


def stun_x2_l_config(**kwargs) -> StunConfig:
    return StunConfig(
        task='x2',
        backbone_name='swin_transformer_224_l_1k',
        pretrained=False,
        in_channels=1536,
        stem_channels=256,
        head_channels=16,
        **kwargs
    )


def stun_x2_l_1k_config(**kwargs) -> StunConfig:
    return StunConfig(
        task='x2',
        backbone_name='swin_transformer_224_l_1k',
        pretrained=True,
        in_channels=1536,
        stem_channels=256,
        head_channels=16,
        **kwargs
    )


def stun_x2_l_22k_config(**kwargs) -> StunConfig:
    return StunConfig(
        task='x2',
        backbone_name='swin_transformer_224_l_22k',
        pretrained=True,
        in_channels=1536,
        stem_channels=256,
        head_channels=16,
        **kwargs
    )
