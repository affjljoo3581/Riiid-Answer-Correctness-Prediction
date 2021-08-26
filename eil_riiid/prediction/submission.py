import argparse
from typing import Dict

import pandas as pd
import torch
import torch.nn as nn

from eil_riiid.data import RiiidVocabulary
from eil_riiid.modeling import CategoricalEmbedding, Transformer
from eil_riiid.prediction.inferencing import RiiidPredictor

try:
    import riiideducation
except ModuleNotFoundError:
    pass


class PredictionModel(nn.Module):
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


def predict_riiid_correctness(args: argparse.ArgumentParser):
    # Prepare a vocabulary of contents and construct the model.
    vocab = RiiidVocabulary(args.vocab_path, use_pad_token=True)

    model = PredictionModel(
        len(vocab),
        args.seq_len,
        vocab.pad_idx,
        args.num_layers,
        args.num_heads,
        args.hidden_dims,
        args.bottleneck,
        dropout_rate=0,
        mask_value=-1e4 if args.use_fp16 else -1e9,
    )
    model.eval().cuda()

    # Load trained model weights.
    model.load_state_dict(torch.load(args.model_path))
    if args.use_fp16:
        model.half()

    # Predict the test data and submit the results.
    predictor = RiiidPredictor(
        vocab, model, args.seq_len, args.max_lag_time, args.max_pqet
    )
    predictor.load_context(args.context_path)

    env = riiideducation.make_env()
    for test_df, _ in env.iter_test():
        env.predict(
            pd.DataFrame(
                predictor.predict(list(test_df.itertuples())),
                columns=["row_id", "answered_correctly"],
            )
        )
