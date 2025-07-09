import os

import torch
from argparse import ArgumentParser
from PIL import Image

from diffusion import DDPM, create_unet
from utils import load_config


def infer_ddpm(config: dict):
    device = config['device']
    epochs = config['num_epoch']
    model_path = os.path.join(config['log_dir'], 'ckpt', f'ckpt_{epochs}.pth')
    n_samples = config['n_samples']
    size = config['sample_size']

    model = create_unet(
        image_size=config['image_size'],
        num_channels=config['num_channels'],
        num_res_blocks=config['num_res_blocks'],
        channel_mult=config['channel_mult']
    )
    model.load_state_dict(torch.load(model_path, weights_only=True)['model'])
    ddpm = DDPM(model, device)
    output = ddpm.infer(n_samples, size, 3)

    # save images
    out_dir = os.path.join(config['log_dir'], f'out_{epochs}_{size[1]}x{size[0]}')
    os.makedirs(out_dir, exist_ok=True)
    for i in range(n_samples):
        img = output[i].cpu().clamp(0, 1).permute(1, 2, 0).numpy()
        img = (img * 255).astype('uint8')
        img = Image.fromarray(img)
        img.save(os.path.join(out_dir, f'sample_{i + 1}.png'))


def main():
    parser = ArgumentParser(description="Sample from the diffusion model.")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the configuration yaml file.")
    args = parser.parse_args()
    infer_ddpm(load_config(args.config))


if __name__ == '__main__':
    main()
