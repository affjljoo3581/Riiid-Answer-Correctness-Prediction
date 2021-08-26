import ast
from collections import defaultdict
from typing import List, Dict, NamedTuple, Tuple, Any

import torch
import torch.nn as nn
import pandas as pd

from eil_riiid.data import RiiidVocabulary


class RiiidPredictor:
    """A stateful predictor of Riiid user-interaction events.

    To predict the answered correctness for Riiid-questions, a model
    needs the contextual history of users. Moreover, feedbacks of the
    prediction should be stored to reuse as the contexts. It provides a
    simple interface to predict the correctness directly without any
    other complex processing. Updating the contexts and gathering the
    history are automatically managed by this class.

    Args:
        vocab: A vocabulary for the user-interaction contents.
        model: An implementation of the prediction model.
        seq_len: A maximum sequence length. The sequence the length of
            which is less than the maximum is padded with the padding
            values.
        max_lag_time: A maximum threshold of lag time for contents.
            All lag times which are greater than the threshold are
            clamped to the threshold value. Default is `1e10`.
        max_pqet: A maximum threshold of prior-question elapsed time.
            All elapsed time which is greater than the threshold is
            clamped to the thresold value. Default is `1e10`.
    """
    def __init__(self,
                 vocab: RiiidVocabulary,
                 model: nn.Module,
                 seq_len: int,
                 max_lag_time: float = 1e10,
                 max_pqet: float = 1e10):
        self.vocab = vocab
        self.model = model

        self.seq_len = seq_len
        self.max_lag_time = max_lag_time
        self.max_pqet = max_pqet

        self.previous_group = None

        self.ctx_table = defaultdict(lambda: defaultdict(list))
        self.latest_correctness = defaultdict(int)
        self.latest_timestamp = defaultdict(float)

    def load_context(self, path: str):
        """Load contexts for user interaction events from file.

        Args:
            path: The user context file path.
        """
        with open(path, 'r') as fp:
            for line in fp:
                data = line.strip().split()

                user_id = int(data[0])
                answered_correctly = list(map(int, data[6].split(',')))

                self.ctx_table[user_id] = {
                    'content_id': list(map(int, data[2].split(','))),
                    'lag_time': list(map(float, data[3].split(','))),
                    'pq_elapsed_time': list(map(float, data[4].split(','))),
                    'pq_had_explanation': list(map(int, data[5].split(','))),
                    'pq_answered_correctly': answered_correctly[:-1]}
                self.latest_correctness[user_id] = answered_correctly[-1]
                self.latest_timestamp[user_id] = float(data[1])

    def _update_context(self, correctness: List[int]):
        """Update the contexts with new interaction events."""
        if self.previous_group is None:
            return

        for x, c in sorted(zip(self.previous_group, correctness),
                           key=lambda x: x[0].timestamp):
            # Extract necessary features from the previous input table.
            embedding = self.vocab.get_embedding_idx(
                x.content_id, x.content_type_id)

            lag_time = (x.timestamp - self.latest_timestamp[x.user_id]) / 1000

            pqet = x.prior_question_elapsed_time
            pqet = 0 if pd.isna(pqet) else pqet

            pqhe = x.prior_question_had_explanation
            pqhe = -1 if pd.isna(pqhe) else (1 if pqhe else 0)

            self.ctx_table[x.user_id]['content_id'].append(embedding)
            self.ctx_table[x.user_id]['lag_time'].append(lag_time)
            self.ctx_table[x.user_id]['pq_elapsed_time'].append(pqet)
            self.ctx_table[x.user_id]['pq_had_explanation'].append(pqhe)
            self.ctx_table[x.user_id]['pq_answered_correctly'].append(
                self.latest_correctness[x.user_id])

            # Resize the context sequence lengths.
            self.ctx_table[x.user_id] = {
                k: v[max(0, len(v) - self.seq_len):]
                for k, v in self.ctx_table[x.user_id].items()}

            # Update the latest correctness and timestamp.
            self.latest_correctness[x.user_id] = c
            self.latest_timestamp[x.user_id] = x.timestamp

    def _preprocess_sequences(self, events: List[Tuple[int, Tuple[Any, ...]]]
                              ) -> Dict[str, torch.Tensor]:
        """Prepare the input tensors for the prediction model."""
        # Define the input data features.
        features = {'content_id': (2, self.vocab.pad_idx, torch.long),
                    'lag_time': (3, 0, torch.float),
                    'pq_elapsed_time': (4, 0, torch.float),
                    'pq_had_explanation': (5, 2, torch.long),
                    'pq_answered_correctly': (6, 2, torch.long)}

        # Parse raw input data to the feature table.
        data = {k: [self.ctx_table[user_id][k] + [x[idx] for x in seq]
                    for user_id, seq in events]
                for k, (idx, _, _) in features.items()}

        # Preprocess the feature values.
        data['lag_time'] = [
            [max(min(x, self.max_lag_time), 0) for x in xs]
            for xs in data['lag_time']]

        data['pq_elapsed_time'] = [
            [max(min(x, self.max_pqet), 0) for x in xs]
            for xs in data['pq_elapsed_time']]

        data['pq_had_explanation'] = [
            [(x if x != -1 else 2) for x in xs]
            for xs in data['pq_had_explanation']]

        data['pq_answered_correctly'] = [
            [(x if x != -1 else 2) for x in xs]
            for xs in data['pq_answered_correctly']]

        # Normalize the lengths of sequences and conver to the
        # `torch.Tensor`s.
        data = {k: [x[max(0, len(x) - self.seq_len):]
                    + [features[k][1]] * (self.seq_len - len(x))
                    for x in seq]
                for k, seq in data.items()}
        data = {k: torch.tensor(v, dtype=features[k][2], device='cuda')
                for k, v in data.items()}

        return data

    def _create_input_data(self, data: List[NamedTuple]
                           ) -> Tuple[Dict[str, torch.Tensor],
                                      Dict[int, Tuple[int, int]]]:
        """Create an input data from the test table."""
        # Sort the interaction events by their timestamp.
        data = sorted(data, key=lambda x: x.timestamp)

        # Group the events by their owners.
        events = defaultdict(list)

        local_latest_timestamp = {}
        for x in data:
            embedding = self.vocab.get_embedding_idx(
                x.content_id, x.content_type_id)

            if x.user_id not in local_latest_timestamp:
                latest = self.latest_timestamp[x.user_id]
            else:
                latest = local_latest_timestamp[x.user_id]

            lag_time = (x.timestamp - latest) / 1000
            local_latest_timestamp[x.user_id] = latest

            pqet = x.prior_question_elapsed_time
            pqet = 0 if pd.isna(pqet) else pqet

            pqhe = x.prior_question_had_explanation
            pqhe = -1 if pd.isna(pqhe) else (1 if pqhe else 0)

            events[x.user_id].append((
                x.row_id, x.content_type_id, embedding, lag_time, pqet, pqhe,
                self.latest_correctness[x.user_id]))
        events = list(events.items())

        # Create a map to indicate the corresponding probability
        # index of each row in table.
        row_maps = {}
        for i, (user_id, xs) in enumerate(events):
            ctx_len = len(self.ctx_table[user_id]['content_id'])
            offset = min(ctx_len + len(xs), self.seq_len) - len(xs)

            for j, x in enumerate(xs):
                if x[1] == 0:
                    row_maps[x[0]] = (i, offset + j)

        return self._preprocess_sequences(events), row_maps

    @torch.no_grad()
    def predict(self, data: List[NamedTuple]) -> List[Tuple[int, float]]:
        """Predict the correctness of questions.

        Args:
            data: An input data table from `DataFrame`.

        Returns:
            A list of predictions containing row ids.
        """
        # Update the context with previous events and their correctness.
        pq_answers_correct = ast.literal_eval(
            data[0].prior_group_answers_correct)

        if pq_answers_correct:
            self._update_context(pq_answers_correct)
        self.previous_group = data

        # Prepare input tensors and predict the correctness.
        x, row_maps = self._create_input_data(data)

        probs = self.model(x).cpu().numpy()
        return [(row_id, probs[x, y]) for row_id, (x, y) in row_maps.items()]
