import torch
import torch.nn as nn
import torch.nn.functional as F


def create_unet_transformer(
    num_channels: int,
    n_classes: int
):
    return UNetTransformerModel(
        n_channel=num_channels,
        N=n_classes
    )


def blk(ic, oc):
    return nn.Sequential(
        nn.Conv2d(ic, oc, 5, padding=2),
        nn.GroupNorm(oc // 8, oc),
        nn.LeakyReLU(),
        nn.Conv2d(oc, oc, 5, padding=2),
        nn.GroupNorm(oc // 8, oc),
        nn.LeakyReLU(),
        nn.Conv2d(oc, oc, 5, padding=2),
        nn.GroupNorm(oc // 8, oc),
        nn.LeakyReLU(),
    )


def blku(ic, oc):
    return nn.Sequential(
        nn.Conv2d(ic, oc, 5, padding=2),
        nn.GroupNorm(oc // 8, oc),
        nn.LeakyReLU(),
        nn.Conv2d(oc, oc, 5, padding=2),
        nn.GroupNorm(oc // 8, oc),
        nn.LeakyReLU(),
        nn.Conv2d(oc, oc, 5, padding=2),
        nn.GroupNorm(oc // 8, oc),
        nn.LeakyReLU(),
        nn.ConvTranspose2d(oc, oc, 2, stride=2),
        nn.GroupNorm(oc // 8, oc),
        nn.LeakyReLU(),
    )


class UNetTransformerModel(nn.Module):

    def __init__(self, n_channel: int, N: int = 16):
        super(UNetTransformerModel, self).__init__()
        self.down1 = blk(n_channel, 16)
        self.down2 = blk(16, 32)
        self.down3 = blk(32, 64)
        self.down4 = blk(64, 512)
        self.down5 = blk(512, 512)
        self.up1 = blku(512, 512)
        self.up2 = blku(512 + 512, 64)
        self.up3 = blku(64, 32)
        self.up4 = blku(32, 16)
        self.convlast = blk(16, 16)
        self.final = nn.Conv2d(16, N * n_channel, 1, bias=False)

        self.tr1 = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        self.tr2 = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        self.tr3 = nn.TransformerEncoderLayer(d_model=64, nhead=8)

        self.temb_1 = nn.Linear(32, 16)
        self.temb_2 = nn.Linear(32, 32)
        self.temb_3 = nn.Linear(32, 64)
        self.temb_4 = nn.Linear(32, 512)
        self.N = N

    def forward(self, x, t) -> torch.Tensor:
        x = (2 * x.float() / self.N) - 1.0   # normalize to [-1, 1]
        t = t.float().reshape(-1, 1) / 1000  # normalize time to [0, 1]
        t_features = ([torch.sin(t * 3.1415 * 2**i) for i in range(16)] +
                      [torch.cos(t * 3.1415 * 2**i) for i in range(16)])
        tx = torch.cat(t_features, dim=1).to(x.device)  # fourier features

        t_emb_1 = self.temb_1(tx).unsqueeze(-1).unsqueeze(-1)
        t_emb_2 = self.temb_2(tx).unsqueeze(-1).unsqueeze(-1)
        t_emb_3 = self.temb_3(tx).unsqueeze(-1).unsqueeze(-1)
        t_emb_4 = self.temb_4(tx).unsqueeze(-1).unsqueeze(-1)

        x1 = self.down1(x) + t_emb_1
        x2 = self.down2(F.avg_pool2d(x1, 2)) + t_emb_2
        x3 = self.down3(F.avg_pool2d(x2, 2)) + t_emb_3
        x4 = self.down4(F.avg_pool2d(x3, 2)) + t_emb_4
        x5 = self.down5(F.avg_pool2d(x4, 2))

        x5 = (
            self.tr1(x5.reshape(x5.shape[0], x5.shape[1], -1).transpose(1, 2))
            .transpose(1, 2)
            .reshape(x5.shape)
        )

        y = self.up1(x5)

        y = (
            self.tr2(y.reshape(y.shape[0], y.shape[1], -1).transpose(1, 2))
            .transpose(1, 2)
            .reshape(y.shape)
        )

        y = self.up2(torch.cat([x4, y], dim=1))

        y = (
            self.tr3(y.reshape(y.shape[0], y.shape[1], -1).transpose(1, 2))
            .transpose(1, 2)
            .reshape(y.shape)
        )
        y = self.up3(y)
        y = self.up4(y)
        y = self.convlast(y)
        y = self.final(y)

        # reshape to (B, C, H, W, N)
        # representing the log probabilities for each class
        y = (
            y.reshape(y.shape[0], -1, self.N, *x.shape[2:])
            .transpose(2, -1)
            .contiguous()
        )

        return y


if __name__ == '__main__':
    from torchinfo import summary
    model = create_unet_transformer(num_channels=3, n_classes=8)
    summary(model, input_data=[torch.randn(16, 3, 64, 64), torch.zeros(16)])
