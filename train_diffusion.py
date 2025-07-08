import os
from argparse import ArgumentParser

import yaml
import torch
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from diffusion import DDPM, create_unet
from utils import load_config, build_batch


def ema_update(target_params: dict, source_params: dict, rate=0.99):
    """
    Update the target parameters using exponential moving average from source parameters.
    """
    for key in target_params.keys():
        target_params[key] = target_params[key] * rate + source_params[key] * (1 - rate)


def train(config: dict):
    device = config['device']
    lr = config['lr']
    num_epoch = config['num_epoch']
    log_dir = config['log_dir']
    ckpt_dir = os.path.join(log_dir, 'ckpt')
    os.makedirs(log_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=log_dir)

    ddpm = DDPM(device)
    batch = build_batch(config)

    model = create_unet(
        image_size=config['image_size'],
        num_channels=config['num_channels'],
        num_res_blocks=config['num_res_blocks'],
        channel_mult=config['channel_mult']
    )
    model.to(device)

    # print and save summary
    stat = summary(model, input_data=[batch, torch.zeros(batch.shape[0], dtype=torch.long)], device=device)
    with open(os.path.join(log_dir, 'model_summary.txt'), 'w') as f:
        f.write(str(stat))

    # save config
    with open(os.path.join(log_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0, total_iters=num_epoch)

    ema_params = {k: v.clone().detach() for k, v in model.state_dict().items()}

    batch_size = config['batch_size']
    mini_batch_size = config['mini_batch_size']
    assert batch_size % mini_batch_size == 0
    optim_step = batch_size // mini_batch_size

    loss_list = []
    for epoch in tqdm(range(num_epoch), desc="Training diffusion"):
        optimizer.zero_grad()
        average_loss = 0
        for mini_step in range(optim_step):
            loss = ddpm.calc_loss(model, batch)
            loss /= optim_step
            average_loss += loss.item()
            loss.backward()
        optimizer.step()
        scheduler.step()

        ema_update(ema_params, model.state_dict())

        if (epoch + 1) % 100 == 0:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            loss_list.append(average_loss)

            writer.add_scalar('loss', average_loss, epoch + 1)
            writer.add_scalar('learning_rate', optimizer.state_dict()['param_groups'][0]['lr'], epoch + 1)
            writer.flush()

        if (epoch + 1) % 1000 == 0:
            save_dict = {
                'model': ema_params,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'loss': average_loss
            }
            torch.save(save_dict, os.path.join(ckpt_dir, f'ckpt_{epoch + 1}.pth'))

    # Save the final model
    if num_epoch % 1000 != 0:
        save_dict ={
            'model': ema_params,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'loss': average_loss
        }
        torch.save(save_dict, os.path.join(ckpt_dir, f'ckpt_{num_epoch}.pth'))
    writer.close()


def main():
    parser = ArgumentParser(description="Train the diffusion model.")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the configuration yaml file.")
    args = parser.parse_args()
    config = load_config(args.config)
    train(config)


if __name__ == '__main__':
    main()
