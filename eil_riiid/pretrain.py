import argparse
from typing import Dict

import stepper
import torch
import torch.nn as nn
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from eil_riiid.data import RiiidDataset, RiiidVocabulary
from eil_riiid.modeling import Transformer


class PretrainingModel(Transformer):
    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        x, _ = super().forward(x["content_id"])
        return self.word_emb(x, transpose=True)


class PretrainingFactory(stepper.Factory):
    def __init__(self, cfg):
        self.cfg = cfg
        self.vocab = RiiidVocabulary(cfg["datasets"]["vocab_path"])

    def create_fetchers(self):
        train_fetcher = RiiidDataset(
            self.cfg["datasets"]["train_dataset_path"],
            self.vocab.pad_idx,
            self.cfg["datasets"]["min_seq_len"],
            self.cfg["datasets"]["max_seq_len"],
        )
        val_fetcher = RiiidDataset(
            self.cfg["datasets"]["val_dataset_path"],
            self.vocab.pad_idx,
            self.cfg["datasets"]["min_seq_len"],
            self.cfg["datasets"]["max_seq_len"],
        )

        return train_fetcher, val_fetcher

    def build_model(self):
        model = PretrainingModel(
            len(self.vocab),
            self.cfg["datasets"]["max_seq_len"],
            self.vocab.pad_idx,
            self.cfg["model"]["num_layers"],
            self.cfg["model"]["num_heads"],
            self.cfg["model"]["hidden_dims"],
            self.cfg["model"]["bottleneck"],
            self.cfg["model"]["dropout_rate"],
            mask_value=-1e4 if self.cfg["training"]["use_amp"] else -1e9,
            bidirectional=False,
        )

        return model

    def configure_optimizer(self, model: nn.Module):
        decay_params, no_decay_params = [], []
        for name, param in model.named_parameters():
            if any(rule in name for rule in ["bias", "LayerNorm.weight"]):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer = AdamW(
            [
                {
                    "params": decay_params,
                    "weight_decay": self.cfg["training"]["weight_decay"],
                },
                {
                    "params": no_decay_params,
                    "weight_decay": 0.0,
                },
            ],
            self.cfg["training"]["base_lr"],
        )
        scheduler = LambdaLR(
            optimizer, lambda x: 1 - x / self.cfg["training"]["total_steps"]
        )

        return optimizer, scheduler


class PretrainingObjective(stepper.Objective):
    def __init__(self, cfg):
        vocab = RiiidVocabulary(cfg["datasets"]["vocab_path"], use_pad_token=True)
        self.criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)

    def train_metrics(
        self, batch: Dict[str, torch.Tensor], model: nn.Module
    ) -> Dict[str, torch.Tensor]:
        return self._calculate_metrics(batch, model)

    def val_metrics(
        self, batch: Dict[str, torch.Tensor], model: nn.Module
    ) -> Dict[str, torch.Tensor]:
        return self._calculate_metrics(batch, model)

    def _calculate_metrics(
        self, batch: Dict[str, torch.Tensor], model: nn.Module
    ) -> Dict[str, torch.Tensor]:
        loss = self.criterion(model(batch).transpose(-1, -2), batch["next_content_id"])
        return {"loss": loss}


def main(args: argparse.Namespace):
    with open(args.config_path, "r") as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)

    # Create callbacks for logging, checkpointing and saving.
    callbacks = [
        stepper.callbacks.TensorBoard(args.log_dir),
        stepper.callbacks.SaveCheckpoint(args.save_ckpt_path),
        stepper.callbacks.SaveLastModel(args.save_last_model_path),
        stepper.callbacks.SaveBestModel(args.save_best_model_path, "loss"),
    ]

    if args.resume_from_ckpt is not None:
        callbacks.append(stepper.callbacks.LoadCheckpoint(args.resume_from_ckpt))

    train_cfg = stepper.TrainConfig(
        cfg["training"]["total_steps"],
        cfg["training"]["val_steps"],
        cfg["training"]["train_batch_size"],
        cfg["training"]["val_batch_size"],
        cfg["datasets"]["num_workers"],
        clip_grad_norm=cfg["training"]["clip_grad_norm"],
        accumulate_grads=cfg["training"]["accumulate_grads"],
        use_amp=cfg["training"]["use_amp"],
        num_gpus=cfg["training"]["num_gpus"],
    )
    stepper.Trainer(
        PretrainingFactory(cfg), PretrainingObjective(cfg), callbacks
    ).train(train_cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="pretrain EiL model")
    parser.add_argument("config_path")
    parser.add_argument("--log_dir", default="training_log")
    parser.add_argument("--save_ckpt_path", default="ckpt.pt")
    parser.add_argument("--save_last_model_path", default="last-model.pt")
    parser.add_argument("--save_best_model_path", default="best-model.pt")
    parser.add_argument("--resume_from_ckpt", default=None)

    main(parser.parse_args())
