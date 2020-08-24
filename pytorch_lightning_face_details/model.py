import difflib
import warnings

import onnxruntime as ort
import pytorch_lightning as pl
import timm
import torch
from torch import nn

from pytorch_lightning_face_details.utils import get_dataloaders

warnings.filterwarnings("ignore")


class FaceDetailsNet(pl.LightningModule):
    def __init__(self, lr=1e-3, backbone="mobilenetv2_100", classes=None):
        self.example_tensor = torch.rand(1, 3, 256, 256)
        super(FaceDetailsNet, self).__init__()
        self.check_backbone(backbone)
        self.model_size = 4
        self.learning_rate = lr
        self.classes = classes

        self.feature_extractor = timm.create_model(backbone, pretrained=True)
        self.dropout = nn.Dropout(0.25)
        self.pretrained_out = nn.Linear(in_features=1000, out_features=6)

    def forward(self, tensor):
        tensor = tensor
        tensor = self.feature_extractor(tensor)
        tensor = self.dropout(tensor)
        tensor = self.pretrained_out(tensor)
        return tensor

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(y_hat, y)

        acc = self.accuracy_score(y, y_hat)
        return {"loss": loss,
                "acc": acc
                }

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss_fn = torch.nn.MSELoss()

        val_loss = loss_fn(y_hat, y)
        val_acc = self.accuracy_score(y, y_hat)
        return {"val_loss": val_loss,
                "val_acc": val_acc
                }

    def training_epoch_end(self, outputs):
        acc = torch.stack([x["acc"] for x in outputs]).mean()

        log = {"acc": acc}
        # log = {}

        return {"log": log, "progress_bar": log}

    def validation_epoch_end(self, outputs):
        val_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        val_acc = torch.stack([x["val_acc"] for x in outputs]).mean()

        log = {"val_loss": val_loss,
               "val_acc": val_acc
               }

        return {"log": log, "progress_bar": log}

    def accuracy_score(self, y, y_hat):
        y = torch.round(y).int()
        y_hat = torch.round(y_hat).int()
        sum_ = torch.sum(y == y_hat)
        accuracy = torch.true_divide(sum_, y.numel())
        return accuracy

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # optimizer = torch.optim.RMSprop(self.parameters(), lr=1e-4)
        return optimizer

    def train_dataloader(self):
        train_dataloader = get_dataloaders()[0]
        return train_dataloader

    def val_dataloader(self):
        val_dataloader = get_dataloaders()[1]
        return val_dataloader

    def decode_data(self, output):
        if not self.classes:
            raise Exception("Please load label classes")
        output = torch.round(output).int()
        result = {self.classes[index].get("name"): self.classes[index].get("classes")[class_number]
                  for index, class_number in enumerate(output.tolist()[0])}
        return result

    def check_backbone(self, backbone):
        model_list = timm.list_models()
        if backbone not in model_list:
            closest_matches = difflib.get_close_matches(backbone, model_list, n=10)
            raise Exception(f"Backbone not in list {model_list}\nClosest matches to {backbone} are: {closest_matches}")

    def save(self, filepath, onnx=False, quantized=False, *args):
        if quantized:
            torch.quantization.quantize_dynamic(self, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8, inplace=True)
            torch.quantization.prepare(self, inplace=True)
            torch.quantization.convert(self, inplace=True)
        if onnx:
            torch.onnx.export(self, self.example_tensor, filepath)
        else:
            model_dict = {"classes": self.classes,
                          "state_dict": self.state_dict()}
            torch.save(model_dict, filepath, *args)

    def load(self, filepath, inference_only=False, **kwargs):
        model_dict = torch.load(filepath, **kwargs)
        self.classes = model_dict.get("classes", None)
        self.load_state_dict(model_dict.get("state_dict", None))

    def predict(self, tensor, decode=False):
        output = self(tensor)
        if decode:
            output = self.decode_data(output)
        return output


if __name__ == "__main__":
    train_dataloader, val_dataloader, classes = get_dataloaders(return_classes=True)

    backbone = "mobilenetv2_100"
    net = FaceDetailsNet(backbone=backbone, classes=classes, lr=1e-3)

    rand_tensor = torch.rand(1, 3, 256, 256)
    t = net(rand_tensor)

    # summary(net, rand_tensor)
    epochs = 15
    trainer = pl.Trainer(min_epochs=1, max_epochs=epochs, gpus=1)
    trainer.fit(net)

    model_name = f"full_model_backbone_{backbone}_epoch_{epochs}_cuda.pth"
    net.save(model_name)
