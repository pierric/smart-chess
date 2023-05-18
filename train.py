import glob
import ffcv
from ffcv.transforms import Convert, ToTensor
from ffcv.fields.decoders import NDArrayDecoder, IntDecoder
import timm
import torch
from torch.optim.lr_scheduler import OneCycleLR
from torch.nn.functional import mse_loss, log_softmax, tanh
from accelerate import Accelerator
from accelerate.utils import LoggerType
from tqdm import tqdm


INIT_LR = 0.001
NUM_EPOCHS = 30

MODEL_VER="v0"
DATASETS = glob.glob(f"{MODEL_VER}/dataset/*.beton")


class ChessModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = timm.create_model('resnet18', pretrained=False, num_classes=0, in_chans=8)

        m = torch.nn.ReLU()

        self.global_pool, self.fc1 = timm.layers.create_classifier(
            self.backbone.num_features,
            4672,
            pool_type="avg",
            use_conv=False,
        )

        self.fc2 = torch.nn.Linear(self.fc1.in_features, 1, bias=True)

    def forward(self, input):
        feature = self.backbone.forward_features(input)

        v = self.global_pool(feature)
        # TODO dropout?
        v1 = self.fc1(v)
        v1 = log_softmax(v1, dim=1)

        v2 = self.fc2(v)
        v2 = tanh(v2)

        return v1, v2


def main():
    model = ChessModel()

    dataloaders = []

    for dataset in DATASETS:
        dl = ffcv.Loader(
            dataset,
            batch_size=32,
            order=ffcv.loader.OrderOption.RANDOM,
            pipelines={
                "board": [NDArrayDecoder(), ToTensor(), Convert(torch.float32)],
                "move": [NDArrayDecoder(), ToTensor()],
                "outcome": [IntDecoder(), ToTensor(), Convert(torch.float32)],
            }
        )
        dataloaders.append(dl)

    num_total = sum(len(dl) for dl in dataloaders)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=INIT_LR, weight_decay=1e-5)

    lr_scheduler = OneCycleLR(
        optimizer=optimizer,
        max_lr=INIT_LR,
        epochs=NUM_EPOCHS,
        steps_per_epoch=num_total
    )

    accelerator = Accelerator(
        log_with="tensorboard",
        project_dir=".",
        mixed_precision="fp16",
        dynamo_backend="INDUCTOR",
    )
    accelerator.init_trackers("logs", config={"init_lr": INIT_LR, "num_epochs": NUM_EPOCHS})

    model, optimizer, lr_scheduler = accelerator.prepare(
        model, optimizer, lr_scheduler
    )

    dataloaders = accelerator.prepare(*dataloaders)

    step = 0
    for epoch in range(NUM_EPOCHS):

        model.train()
        accelerator.print(f"epoch {epoch}")

        with tqdm(total=num_total) as pbar:
            for dl in dataloaders:
                for batch in dl:
                    batch = [v.to(accelerator.device) for v in batch]
                    output_distr, output_award = model(batch[0])

                    loss1 = - torch.sum(output_distr * batch[1]) / output_distr.size()[0]
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
    accelerator.save_state(output_dir=f"{MODEL_VER}/checkpoint")


if __name__ == "__main__":
    main()
