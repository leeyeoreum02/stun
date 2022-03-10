__all__ = ['StunConfig', 'StunX4TConfig']


class StunConfig:
    def __init__(
        self,
        task: str,
        backbone_name: str,
        in_channels: int,
        head_channels: int,
        **kwargs
    ) -> None:
        self.task = task
        self.in_channels = in_channels
        self.backbone_name = backbone_name
        self.head_channels = head_channels


class StunX4TConfig(StunConfig):
    def __init__(self, **kwargs) -> None:
        super(StunX4TConfig, self).__init__(
            task='x4',
            backbone_name='swin_transformer_96_t',
            in_channels=768,
            head_channels=64,
            **kwargs
        )
