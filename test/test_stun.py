import torch
from torchinfo import summary

from stun.model.configuration import StunX4TConfig
from stun.model.stun import Stun


def main():
    cfg = StunX4TConfig()
    model = Stun(cfg)
    # print(summary(model, input_size=(1, 3, 96, 96)))

    noise_x4 = torch.randn(1, 3, 96, 96)

    backbone_output = model.backbone(noise_x4)
    print(f'backbone_output.shape: {backbone_output.shape}, backbone_output.type(): {backbone_output.type()}')
    output_stunx4_t = model(noise_x4)
    print(f'output_swinx4_t.shape: {output_stunx4_t.shape}, output_swinx4_t.type(): {output_stunx4_t.type()}')


if __name__ == '__main__':
    main()
