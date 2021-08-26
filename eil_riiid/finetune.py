import argparse
from typing import Dict

import numpy as np
import stepper
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import roc_auc_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from eil_riiid.data import RiiidDataset, RiiidVocabulary
from eil_riiid.modeling import CategoricalEmbedding, Transformer


class FinetuningModel(nn.Module):
    def __init__(
        self,
        num_words: int,
        seq_len: int,
        pad_idx: int,
        max_lag_time: int,
        max_pqet: int,
        num_layers: int,
        num_heads: int,
        hidden_dims: int,
        bottleneck: int = 4,
        dropout_rate: float = 0.1,
        mask_value: float = -1e9,
    ):
        super().__init__()
        self.auxiliary_embs = nn.ModuleDict(
            {
                "lag_time": CategoricalEmbedding(max_lag_time, hidden_dims),
                "pq_elapsed_time": CategoricalEmbedding(max_pqet, hidden_dims),
                "pq_had_explanation": CategoricalEmbedding(3, hidden_dims),
                "pq_answered_correctly": CategoricalEmbedding(3, hidden_dims),
            }
        )
        self.transformer = Transformer(
            num_words,
            seq_len,
            pad_idx,
            num_layers,
            num_heads,
            hidden_dims,
            bottleneck,
            dropout_rate,
            mask_value,
            bidirectional=False,
        )
        self.proj_head = nn.Linear(hidden_dims, 1)

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        aux = 0
        for key, embedding_layer in self.auxiliary_embs.items():
            aux += embedding_layer(x[key])

        x, _ = self.transformer(x["content_id"], aux)
        x = self.proj_head(x).squeeze(-1)

        return x


class FinetuningFactory(stepper.Factory):
    def __init__(self, cfg, pretrained_model_path):
        self.cfg = cfg
        self.vocab = RiiidVocabulary(cfg["datasets"]["vocab_path"])
        self.pretrained_model_path = pretrained_model_path

    def create_fetchers(self):
        train_fetcher = RiiidDataset(
            self.cfg["datasets"]["train_dataset_path"],
            self.vocab.pad_idx,
            0,
            self.cfg["datasets"]["seq_len"],
            self.cfg["datasets"]["max_lag_time"],
            self.cfg["datasets"]["max_pqet"],
        )
        val_fetcher = RiiidDataset(
            self.cfg["datasets"]["val_dataset_path"],
            self.vocab.pad_idx,
            0,
            self.cfg["datasets"]["seq_len"],
            self.cfg["datasets"]["max_lag_time"],
            self.cfg["datasets"]["max_pqet"],
        )

        return train_fetcher, val_fetcher

    def build_model(self):
        model = FinetuningModel(
            len(self.vocab),
            self.cfg["datasets"]["seq_len"],
            self.vocab.pad_idx,
            self.cfg["datasets"]["max_lag_time"],
            self.cfg["datasets"]["max_pqet"],
            self.cfg["model"]["num_layers"],
            self.cfg["model"]["num_heads"],
            self.cfg["model"]["hidden_dims"],
            self.cfg["model"]["bottleneck"],
            self.cfg["model"]["dropout_rate"],
            mask_value=-1e4 if self.cfg["training"]["use_amp"] else -1e9,
        )

        if self.pretrained_model_path is not None:
            model.transformer.load_state_dict(
                torch.load(self.pretrained_model_path, map_location="cpu")
            )

        return model

    def configure_optimizer(self, model: nn.Module):
        decay_params, no_decay_params = [], []
        for name, param in model.named_parameters():
            if any(rule in name for rule in ["bias", "ln_"]):
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


class FinetuningObjective(stepper.Objective):
    def __init__(self, cfg):
        self.positive_ratio = cfg["datasets"]["positive_ratio"]
        self.vocab = RiiidVocabulary(cfg["datasets"]["vocab_path"], use_pad_token=True)

    def train_metrics(
        self, batch_inputs: Dict[str, torch.Tensor], model: nn.Module
    ) -> Dict[str, torch.Tensor]:
        return self._calculate_metrics(batch_inputs, model)

    def val_metrics(
        self, batch_inputs: Dict[str, torch.Tensor], model: nn.Module
    ) -> Dict[str, torch.Tensor]:
        return self._calculate_metrics(batch_inputs, model)

    def _calculate_metrics(
        self, batch_inputs: Dict[str, torch.Tensor], model: nn.Module
    ) -> Dict[str, torch.Tensor]:
        logits = model(batch_inputs)
        loss = self._calculate_loss(batch_inputs, logits)
        roc_auc = self._calculate_roc_auc(batch_inputs, logits)

        return {"loss": loss, "roc_auc": roc_auc}

    def _calculate_loss(
        self, x: Dict[str, torch.Tensor], logits: torch.Tensor
    ) -> torch.Tensor:
        # In binary classification task, class imbalance can be occured.
        # To resolve that, we are going to apply class-weights to the
        # loss.
        pos = (x["answered_correctly"] == 1).type_as(logits)
        weights = pos * (1 / self.positive_ratio - 1) + (1 - pos) * 1

        # Futhermore, the non-answerable contents are in the labels.
        # They should be ignored in calculating the loss.
        mask = (x["answered_correctly"] != -1).type_as(logits)

        loss = F.binary_cross_entropy_with_logits(
            logits, x["answered_correctly"].type_as(logits), reduction="none"
        )
        return (mask * weights * loss).sum() / mask.sum()

    def _calculate_roc_auc(
        self, x: Dict[str, torch.Tensor], logits: torch.Tensor
    ) -> torch.Tensor:
        # To calculate AUC score of ROC curve, the true labels and
        # prediction probabilities are required. Since the tensors are
        # in GPU memory, we need to move the tensors to CPU memory.
        y_true = x["answered_correctly"].detach().cpu().numpy()
        y_score = torch.sigmoid(logits.float()).detach().cpu().numpy()

        # While the prediction is performed only with the last elements,
        # the ROC AUC score must be calculated for the last non-padded
        # elements.
        tokens = x["content_id"].detach().cpu().numpy()

        is_pad = tokens == self.vocab.pad_idx
        is_pad = np.pad(is_pad, ((0, 0), (0, 1)), constant_values=True)
        mask = (is_pad[:, 1:] ^ is_pad[:, :-1]) & (y_true != -1)

        try:
            return torch.tensor(
                roc_auc_score(y_true[mask], y_score[mask]), device="cuda"
            )
        except ValueError:
            return 0


def main(args: argparse.Namespace):
    with open(args.config_path, "r") as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)

    # Create callbacks for logging, checkpointing and saving.
    callbacks = [
        stepper.callbacks.TensorBoard(args.log_dir),
        stepper.callbacks.SaveCheckpoint(args.save_ckpt_path),
        stepper.callbacks.SaveBestModel(
            args.save_model_path, monitor_metric="roc_auc", mode="max"
        ),
    ]

    if cfg["training"]["early_stop"]:
        callbacks.append(
            stepper.callbacks.EarlyStopping(
                monitor_metric="roc_auc", patience=3, mode="max"
            )
        )

    if args.resume_from_ckpt is not None:
        callbacks.append(stepper.callbacks.LoadCheckpoint(args.resume_from_ckpt))

    train_cfg = stepper.TrainConfig(
        cfg["training"]["total_steps"],
        cfg["training"]["val_steps"],
        cfg["training"]["train_batch_size"],
        cfg["training"]["val_batch_size"],
        cfg["datasets"]["num_workers"],
        clip_grad_norm=cfg["training"]["clip_grad_norm"],
        use_amp=cfg["training"]["use_amp"],
        num_gpus=cfg["training"]["num_gpus"],
    )
    stepper.Trainer(
        FinetuningFactory(cfg, args.pretrained_model_path),
        FinetuningObjective(cfg),
        callbacks,
    ).train(train_cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="pretrain EiL model")
    parser.add_argument("config_path")
    parser.add_argument("--log_dir", default="training_log")
    parser.add_argument("--save_ckpt_path", default="ckpt.pt")
    parser.add_argument("--save_model_path", default="model.pt")
    parser.add_argument("--pretrained_model_path", default="pretrained.pt")
    parser.add_argument("--resume_from_ckpt", default=None)

    main(parser.parse_args())
