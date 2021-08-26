from typing import Tuple


class RiiidVocabulary:
    """A vocabulary for contents in Riiid user-interaction events.

    Args:
        path: The vocabulary file path.
        use_pad_token: A boolean determining whether to use a padding token to the
            vocabulary. Default is `True`.
    """

    def __init__(self, path: str, use_pad_token: bool = True):
        with open(path, "r") as fp:
            self.contents = [tuple(int(x) for x in y.strip().split()) for y in fp]

        # Add the padding token to the table.
        if use_pad_token:
            self.contents.append("[PAD]")

        self.lookup_table = {k: v for v, k in enumerate(self.contents)}

    def __len__(self) -> int:
        """Return the number of contents."""
        return len(self.contents)

    def __getitem__(self, embedding_idx: int) -> Tuple[int, int]:
        """Return the content id and type."""
        return self.contents[embedding_idx]

    def get_embedding_idx(self, content_id: int, content_type: int) -> int:
        """Return the embedding index."""
        return self.lookup_table[(content_id, content_type)]

    @property
    def pad_idx(self) -> int:
        """Return the embedding index of the padding token."""
        return self.lookup_table["[PAD]"]
