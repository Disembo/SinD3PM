import os

import torch
from argparse import ArgumentParser

import torchvision
from PIL import Image

from diffusion import D3PM, create_unet_categorical
from utils import load_config


def infer_d3pm(config: dict):
    N = config['num_classes']
    device = config['device']
    epochs = config['num_epoch']
    model_path = os.path.join(config['log_dir'], 'ckpt', f'ckpt_{epochs}.pth')
    n_samples = config['n_samples']
    size = config['sample_size']
    out_dir = os.path.join(config['log_dir'], f'out_{epochs}_{size[1]}x{size[0]}')
    os.makedirs(out_dir, exist_ok=True)

    model = create_unet_categorical(
        image_size=config['image_size'],
        num_classes=N,
        num_channels=config['num_channels'],
        num_res_blocks=config['num_res_blocks'],
        channel_mult=config['channel_mult']
    )
    model.load_state_dict(torch.load(model_path, weights_only=True))
    d3pm = D3PM(model, device, forward_type=config['forward_type'], n_steps=1000, num_classes=N, hybrid_loss_coef=0.0)

    # image sequences to gif
    seq = d3pm.infer_with_image_sequence(n_samples, size, 3, stride=40)
    gif = []
    for batch in seq:
        img = torchvision.utils.make_grid(batch.float() / (N - 1), nrow=2)
        img = img.permute(1, 2, 0).cpu().numpy()
        img = (img * 255).astype('uint8')
        gif.append(Image.fromarray(img))

    gif[0].save(
        os.path.join(out_dir, f'sample.gif'),
        save_all=True,
        append_images=gif[1:],
        duration=100,
        loop=0,
    )
    gif[-1].save(os.path.join(out_dir, f'sample_last.png'))


def main():
    parser = ArgumentParser(description="Sample from the diffusion model.")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the configuration yaml file.")
    args = parser.parse_args()
    infer_d3pm(load_config(args.config))


if __name__ == '__main__':
    main()
