import os
from argparse import ArgumentParser

import torch
import torchvision
import yaml
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from tqdm import tqdm

from diffusion import D3PM, create_unet_categorical
from utils import load_config, build_batch


def train_d3pm(config: dict):
    device = config['device']
    lr = config['lr']
    num_epoch = config['num_epoch']
    log_dir = config['log_dir']
    ckpt_dir = os.path.join(log_dir, 'ckpt')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=log_dir)

    N = config['num_classes']
    model = create_unet_categorical(
        image_size=config['image_size'],
        num_classes=N,
        num_channels=config['num_channels'],
        num_res_blocks=config['num_res_blocks'],
        channel_mult=config['channel_mult']
    )
    d3pm = D3PM(model, device, forward_type=config['forward_type'], n_steps=1000, num_classes=N, hybrid_loss_coef=0.0)
    batch = build_batch(config).to(device)
    batch = (batch * (N - 1)).round().long().clamp(0, N - 1)  # discretize to N bins

    # save gt image
    torchvision.utils.save_image(
        [batch[0].float()],
        os.path.join(log_dir, 'gt_image.png'),
        nrow=2,
        normalize=True,
        value_range=(0, N - 1)
    )

    # print and save summary
    stat = summary(model, input_data=[batch, torch.zeros(batch.shape[0], dtype=torch.long)], device=device)
    with open(os.path.join(log_dir, 'model_summary.txt'), 'w') as f:
        f.write(str(stat))

    # save config
    with open(os.path.join(log_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0, total_iters=num_epoch)

    batch_size = config['batch_size']
    mini_batch_size = config['mini_batch_size']
    assert batch_size % mini_batch_size == 0
    optim_step = batch_size // mini_batch_size

    for epoch in tqdm(range(num_epoch)):
        model.train()
        average_loss = 0.0
        for mini_step in range(optim_step):
            optimizer.zero_grad()
            loss, info = d3pm.calc_loss(batch)
            loss /= optim_step
            average_loss += loss.item()
            loss.backward()
        optimizer.step()
        scheduler.step()

        norm = torch.nn.utils.clip_grad_norm_(d3pm.x0_model.parameters(), 0.1)

        with torch.no_grad():
            param_norm = sum([torch.norm(p) for p in d3pm.x0_model.parameters()])

        if (epoch + 1) % config['log_every'] == 0:
            writer.add_scalar('loss', average_loss, epoch + 1)
            writer.add_scalar('norm', norm, epoch + 1)
            writer.add_scalar('param_norm', param_norm, epoch + 1)
            writer.add_scalar('vb_loss', info['vb_loss'], epoch + 1)
            writer.add_scalar('ce_loss', info['ce_loss'], epoch + 1)
            writer.add_scalar('learning_rate', optimizer.state_dict()['param_groups'][0]['lr'], epoch + 1)
            writer.flush()

        if (epoch + 1) % config['save_every'] == 0:
            torch.save(model.state_dict(), os.path.join(ckpt_dir, f'ckpt_{epoch + 1}.pth'))

    # Save the final model
    if num_epoch % config['save_every'] != 0:
        torch.save(model.state_dict(), os.path.join(ckpt_dir, f'ckpt_{num_epoch}.pth'))
    writer.close()


def main():
    parser = ArgumentParser(description="Train the diffusion model.")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the configuration yaml file.")
    args = parser.parse_args()
    config = load_config(args.config)
    train_d3pm(config)


if __name__ == '__main__':
    main()
