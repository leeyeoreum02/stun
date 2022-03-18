import torch
from stun.model.feature_extraction import swin_transformer_160_t, swin_transformer_96_t, swin_transformer_96_s
from stun.model.feature_extraction import swin_transformer_96_b, swin_transformer_96_l
from stun.model.feature_extraction import swin_transformer_224_t, swin_transformer_224_s


def main():
    noise_x2 = torch.randn(1, 3, 224, 224)
    # noise_x4 = torch.randn(1, 3, 96, 96)
    noise_x4_160 = torch.randn(1, 3, 160, 160)

    swinx2_t = swin_transformer_224_t()
    swinx2_s = swin_transformer_224_s()
    swinx4_t = swin_transformer_96_t()
    swinx4_s = swin_transformer_96_s()
    swinx4_b = swin_transformer_96_b()
    swinx4_l = swin_transformer_96_l()
    swin_x4_160_t = swin_transformer_160_t()

    output_swinx2_t = swinx2_t(noise_x2)
    print(f'output_swinx2_t.shape: {output_swinx2_t.shape}, output_swinx2_t.type(): {output_swinx2_t.type()}')

    # output_swinx2_s = swinx2_s(noise_x2)
    # print(f'output_swinx2_s.shape: {output_swinx2_s.shape}, output_swinx2_s.type(): {output_swinx2_s.type()}')

    # output_swinx2_b = swinx2_b(noise_x2)
    # print(f'output_swinx2_b.shape: {output_swinx2_b.shape}, output_swinx2_b.type(): {output_swinx2_b.type()}')

    # output_swinx2_l = swinx2_l(noise_x2)
    # print(f'output_swinx2_l.shape: {output_swinx2_l.shape}, output_swinx2_l.type(): {output_swinx2_l.type()}')

    # output_swinx4_t = swinx4_t(noise_x4)
    # print(f'output_swinx4_t.shape: {output_swinx4_t.shape}, output_swinx4_t.type(): {output_swinx4_t.type()}')

    # output_swinx4_s = swinx4_s(noise_x4)
    # print(f'output_swinx4_s.shape: {output_swinx4_s.shape}, output_swinx4_s.type(): {output_swinx4_s.type()}')

    # output_swinx4_b = swinx4_b(noise_x4)
    # print(f'output_swinx4_b.shape: {output_swinx4_b.shape}, output_swinx4_b.type(): {output_swinx4_b.type()}')

    # output_swinx4_l = swinx4_l(noise_x4)
    # print(f'output_swinx4_l.shape: {output_swinx4_l.shape}, output_swinx4_l.type(): {output_swinx4_l.type()}')
    
    output_swinx4_b = swin_x4_160_t(noise_x4_160)
    print(f'output_swinx4_b.shape: {output_swinx4_b.shape}, output_swinx4_b.type(): {output_swinx4_b.type()}')


if __name__ == '__main__':
    main()
