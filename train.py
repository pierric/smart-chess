import glob
import ffcv
from ffcv.transforms import Convert, ToTensor, ToTorchImage
from ffcv.fields.decoders import NDArrayDecoder, IntDecoder
#import timm
from torchvision.models import vgg11
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import OneCycleLR
from torch.nn.functional import mse_loss, log_softmax, tanh, relu
from accelerate import Accelerator
from accelerate.utils import LoggerType
from tqdm import tqdm


class ResBlock(nn.Module):
    def __init__(self, inplanes=256, planes=256, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = relu(self.bn1(out))
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = relu(out)
        return out


class ChessModel(torch.nn.Module):
    N_RES_BLOCKS = 1

    def __init__(self):
        super().__init__()

        # 8 boards (14 channels each) + meta (7 channels)
        self.conv_block = nn.Sequential(
            nn.Conv2d(14 * 8 + 7, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.res_blocks = nn.ModuleList([ResBlock()] * self.N_RES_BLOCKS )

        self.value_head = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

        self.policy_head = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(8*8*128, 8*8*73),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, inp):
        x = self.conv_block(inp)

        for block in self.res_blocks:
            x = block(x)

        v1 = self.policy_head(x).exp()
        v2 = self.value_head(x)
        return v1, v2


@click.command()
@click.option("--init-lr", default=0.001)
@click.option("--num-epochs", default=30)
@click.option("--model-ver", default=0)
def main(init_lr, num_epochs, model_ver):
    model = ChessModel()

    dataloaders = []

    datasets = glob.glob(f"v{model_ver}/dataset/*.beton")

    for dataset in datasets:
        dl = ffcv.Loader(
            dataset,
            batch_size=32,
            order=ffcv.loader.OrderOption.RANDOM,
            pipelines={
                "board": [NDArrayDecoder(), ToTorchImage(), Convert(torch.float32)],
                "move": [NDArrayDecoder(), ToTensor()],
                "outcome": [IntDecoder(), ToTensor(), Convert(torch.float32)],
            }
        )
        dataloaders.append(dl)

    num_total = sum(len(dl) for dl in dataloaders)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=init_lr, weight_decay=1e-5)

    lr_scheduler = OneCycleLR(
        optimizer=optimizer,
        max_lr=init_lr,
        epochs=num_epochs,
        steps_per_epoch=num_total
    )

    accelerator = Accelerator(
        log_with="tensorboard",
        project_dir=".",
        mixed_precision="fp16",
        dynamo_backend="INDUCTOR",
    )
    accelerator.init_trackers("logs", config={"init_lr": init_lr, "num_epochs": num_epochs})

    model, optimizer, lr_scheduler = accelerator.prepare(
        model, optimizer, lr_scheduler
    )

    dataloaders = accelerator.prepare(*dataloaders)

    if model_ver >= 1:
        accelerator.load_state(f"v{model_ver-1}/checkpoint")

    step = 0
    for epoch in range(num_epochs):

        model.train()
        accelerator.print(f"epoch {epoch}")

        with tqdm(total=num_total) as pbar:
            for dl in dataloaders:
                for batch in dl:
                    batch = [v.to(accelerator.device) for v in batch]
                    output_distr, output_award = model(batch[0])

                    loss1 = - torch.sum(output_distr.log() * batch[1], dim=1).mean()
                    loss2 = mse_loss(output_award, batch[2])

                    loss = loss1 + loss2
                    accelerator.backward(loss)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    loss = loss.detach().cpu().numpy()
                    lr = lr_scheduler.get_last_lr()[0]
                    accelerator.log(
                        {"lr": lr, "train_loss": loss.item()},
                        step=step
                    )
                    step+=1
                    pbar.update()

    accelerator.end_training()
    accelerator.save_state(output_dir=f"v{model_ver}/checkpoint")


if __name__ == "__main__":
    main()
