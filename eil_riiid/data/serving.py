import warnings
from typing import Any, Dict, Optional

import stepper
import torch

warnings.filterwarnings(action="ignore")


class RiiidDataset(stepper.DataFetcher):
    """A data fetcher of Riiid user-interaction events.

    The user-interaction events are preprocessed from the original dataset of the
    competition `Riiid! Answer Correctness Prediction`. This class fetches the event
    data and pads the sequences with proper padding values.

    Args:
        path: The preprocessed dataset path.
        pad_idx: The embedding index of padding token for contents.
        min_seq_len: A minimum sequence length. The sequence the length of which is less
            than the minimum is ignored.
        max_seq_len: A maximum sequence length. The sequence the length of which is less
            than the maximum is padded with the padding values.
        max_lag_time: A maximum threshold of lag time for contents. All lag times which
            are greater than the threshold are clamped to the threshold value. Default
            is `1e10`.
        max_pqet: A maximum threshold of prior-question elapsed time. All elapsed times
            which are greater than the threshold are clamped to the threshold value.
            Default is `1e10`.
    """

    def __init__(
        self,
        path: str,
        pad_idx: int,
        min_seq_len: int,
        max_seq_len: int,
        max_lag_time: float = 1e10,
        max_pqet: float = 1e10,
    ):
        self.pad_idx = pad_idx
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.max_lag_time = max_lag_time
        self.max_pqet = max_pqet

        self.dataset_fp = open(path, "r")
        self.i = 0

    def __del__(self):
        try:
            self.dataset_fp.close()
        except Exception:
            pass

    def _prepare_data_item(self, data: str) -> Dict[str, torch.Tensor]:
        # Define the input data features.
        features = [
            ("content_id", int, torch.long, self.pad_idx),
            ("lag_time", float, torch.long, 0),
            ("pq_elapsed_time", float, torch.long, 0),
            ("pq_had_explanation", int, torch.long, 2),
            ("pq_answered_correctly", int, torch.long, 2),
            ("next_content_id", int, torch.long, self.pad_idx),
            ("answered_correctly", int, torch.long, -1),
        ]

        # Parse comma-separated sequences to the features.
        data = data.strip().split()
        data = {
            name: list(map(dtype, data[i].split(",")))
            for i, (name, dtype, _, _) in enumerate(features)
        }

        # Skip sequences the lengths of which are too short.
        if len(data["content_id"]) < self.min_seq_len:
            return None

        # Preprocess the feature values.
        data["lag_time"] = list(
            map(lambda x: max(min(x, self.max_lag_time - 1), 0), data["lag_time"])
        )
        data["pq_elapsed_time"] = list(
            map(lambda x: max(min(x, self.max_pqet - 1), 0), data["pq_elapsed_time"])
        )
        data["pq_had_explanation"] = list(
            map(lambda x: x if x != -1 else 2, data["pq_had_explanation"])
        )
        data["pq_answered_correctly"] = list(
            map(lambda x: x if x != -1 else 2, data["pq_answered_correctly"])
        )
        data["next_content_id"] = list(
            map(lambda x: x if x != -1 else self.pad_idx, data["next_content_id"])
        )

        # Pad the sequences and convert to the `torch.Tensor`s.
        data = {
            name: data[name] + [pad] * (self.max_seq_len - len(data[name]))
            for name, _, _, pad in features
        }
        data = {
            name: torch.tensor(data[name], dtype=dtype)
            for name, _, dtype, _ in features
        }

        return data

    def fetch(self) -> Optional[Dict[str, torch.Tensor]]:
        while True:
            data = self.dataset_fp.readline()
            if not data:
                raise StopIteration()

            data = self._prepare_data_item(data)
            if data is not None:
                return data

    def skip(self, after: int):
        while after > 0:
            data = self.dataset_fp.readline()
            if not data:
                raise StopIteration()

            seq_len = len(data.strip().split()[0].split(","))
            if seq_len < self.min_seq_len:
                continue

            after -= 1

    def state_dict(self) -> Dict[str, Any]:
        return {"offset": self.dataset_fp.tell()}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.dataset_fp.seek(state_dict["offset"])
