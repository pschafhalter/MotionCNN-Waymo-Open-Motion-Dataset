import numpy as np
import torch
import timm
import os
import argparse
from glob import glob
from tqdm import tqdm
from utils import get_config
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from losses import NLLGaussian2d
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


class DistributedModel(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = get_model(config["model"])
        self.loss_module = NLLGaussian2d()

    def forward(self, batch, batch_idx=None):
        return self.model(batch)

    def configure_optimizers(self):
        result = dict(
            optimizer=AdamW(
                self.model.parameters(), **self.config["training"]["optimizer"]
            )
        )
        scheduler_config = self.config["training"].get("scheduler")
        if scheduler_config is not None:
            result["scheduler"] = CosineAnnealingWarmRestarts(
                result["optimizer"], **scheduler_config
            )
        return result

    def common_step(self, batch, batch_idx):
        batch = dict_to_cuda(batch)
        pd_tensor = self.forward(batch["raster"], batch_idx)
        pd_dict = postprocess_predictions(pd_tensor, self.config["model"])
        loss = self.loss_module(batch, pd_dict)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        self.log("validation/loss", loss, sync_dist=True)
        return loss


class MotionCNNDataset(Dataset):
    def __init__(self, data_path, load_roadgraph=False) -> None:
        super().__init__()
        self._load_roadgraph = load_roadgraph
        self._files = glob(os.path.join(data_path, "*", "agent_data", "*.npz"))
        self._roadgraph_data = glob(
            os.path.join(data_path, "*", "roadgraph_data", "segments_global.npz")
        )
        self._scid_to_roadgraph = {f.split("/")[-3]: f for f in self._roadgraph_data}

    def __len__(self):
        return len(self._files)

    def __getitem__(self, idx):
        data = dict(np.load(self._files[idx], allow_pickle=True))
        if self._load_roadgraph:
            roadgraph_data_file = self._scid_to_roadgraph[data["scenario_id"].item()]
            roadgraph_data = np.load(roadgraph_data_file)["roadgraph_segments"]
            roadgraph_valid = np.ones(roadgraph_data.shape[0])
            n_to_pad = 6000 - roadgraph_data.shape[0]
            roadgraph_data = np.pad(roadgraph_data, ((0, n_to_pad), (0, 0), (0, 0)))
            roadgraph_valid = np.pad(roadgraph_valid, (0, n_to_pad))
            data["roadgraph_data"] = roadgraph_data
            data["roadgraph_valid"] = roadgraph_valid
        data["raster"] = data["raster"].transpose(2, 0, 1).astype(np.float32) / 255.0
        data["scenario_id"] = data["scenario_id"].item()
        data["future_valid"] = data["future_valid"].astype(np.int32)
        return data


def dict_to_cuda(data_dict):
    gpu_required_keys = ["raster", "future_valid", "future_local"]
    data_dict["raster"] = data_dict["raster"].float()
    data_dict["future_valid"] = data_dict["future_valid"].int()
    for key in gpu_required_keys:
        data_dict[key] = data_dict[key].cuda(non_blocking=True)
    return data_dict


def get_model(model_config):
    # x, y, sigma_xx, sigma_yy, visibility
    n_components = 5
    n_modes = model_config["n_modes"]
    n_timestamps = model_config["n_timestamps"]
    output_dim = n_modes + n_modes * n_timestamps * n_components
    model = timm.create_model(
        model_config["backbone"], pretrained=True, in_chans=27, num_classes=output_dim
    )
    return model


def limited_softplus(x):
    return torch.clamp(F.softplus(x), min=0.1, max=10)


def postprocess_predictions(predicted_tensor, model_config):
    confidences = predicted_tensor[:, : model_config["n_modes"]]
    components = predicted_tensor[:, model_config["n_modes"] :]
    components = components.reshape(
        -1, model_config["n_modes"], model_config["n_timestamps"], 5
    )
    sigma_xx = components[:, :, :, 2:3]
    sigma_yy = components[:, :, :, 3:4]
    visibility = components[:, :, :, 4:]
    return {
        "confidences": confidences,
        "xy": components[:, :, :, :2],
        "sigma_xx": limited_softplus(sigma_xx)
        if model_config["predict_covariances"]
        else torch.ones_like(sigma_xx),
        "sigma_yy": limited_softplus(sigma_yy)
        if model_config["predict_covariances"]
        else torch.ones_like(sigma_yy),
        "visibility": visibility,
    }


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-data-path", type=str, required=True, help="Path to training data"
    )
    parser.add_argument(
        "--val-data-path", type=str, required=True, help="Path to validation data"
    )
    parser.add_argument(
        "--checkpoint-path", type=str, required=False, help="Path to checkpoint"
    )
    parser.add_argument("--config", type=str, required=True, help="Config file path")
    args = parser.parse_args()
    return args


def get_last_checkpoint_file(path):
    list_of_files = glob(f"{path}/*.pth")
    if len(list_of_files) == 0:
        return None
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file


def main():
    args = parse_arguments()
    general_config = get_config(args.config)
    model_config = general_config["model"]
    training_config = general_config["training"]
    config_name = args.config.split("/")[-1].split(".")[0]
    model = DistributedModel(general_config)

    training_dataloader = DataLoader(
        MotionCNNDataset(args.train_data_path), **training_config["train_dataloader"]
    )
    validation_dataloader = DataLoader(
        MotionCNNDataset(args.val_data_path, load_roadgraph=False),
        **training_config["val_dataloader"],
    )

    logger = TensorBoardLogger(
        "models", f"MotionCNN-{model_config['backbone']}-waymo-open-dataset"
    )
    callbacks = [ModelCheckpoint(every_n_epochs=1, save_top_k=-1)]
    trainer = Trainer(
        max_epochs=training_config["num_epochs"],
        callbacks=callbacks,
        logger=logger,
        strategy="ddp",
    )

    if args.checkpoint_path:
        trainer.fit(
            model,
            training_dataloader,
            validation_dataloader,
            ckpt_path=args.checkpoint_path,
        )
    else:
        trainer.fit(model, training_dataloader, validation_dataloader)


if __name__ == "__main__":
    main()
