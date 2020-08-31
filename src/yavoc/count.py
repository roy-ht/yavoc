import collections
import operator
import typing as T

from .base import Vocabulary


class CountVocabulary(Vocabulary[int]):
    """Vocabulary from token counts"""

    def __init__(
        self,
        sentences: T.Optional[T.Iterable[T.Iterable[str]]] = None,
        min_count: int = 1,
        max_vocab_size: T.Optional[int] = None,
        padding: bool = True,
        padding_token: T.Optional[str] = None,
        oov_token: T.Optional[str] = None,
    ) -> None:
        super().__init__(padding=padding, padding_token=padding_token, oov_token=oov_token)
        self._min_count = min_count
        self._max_vocab_size = max_vocab_size
        self._counter: T.Optional[collections.Counter] = None
        self._index_to_token: T.List[str] = []

        if sentences is not None:
            self.update(sentences)

    @property
    def index_to_token_dict(self) -> T.List[str]:
        return self._index_to_token

    @property
    def counter_dict(self):
        return self._counter

    @property
    def min_count(self) -> int:
        return self._min_count

    @min_count.setter
    def min_count(self, v: int):
        if v != self._min_count:
            self._min_count = v
            self.build()

    @property
    def max_vocab_size(self):
        return self._max_vocab_size

    @max_vocab_size.setter
    def max_vocab_size(self, v: T.Optional[int]):
        if v != self._max_vocab_size:
            self._max_vocab_size = v
            self.build()

    def to_tokens(self, id_sentences: T.Iterable[T.Iterable[int]], remove_paddings: bool = True) -> T.List[T.List[str]]:
        if not self._index_to_token:
            self._build_index_to_token()
        retlist = []
        padding_idx = self._token_to_index.get(self._padding_token)
        for ids in id_sentences:
            tokens = []
            for idx in ids:
                if remove_paddings and padding_idx is not None and idx == padding_idx:
                    continue
                tokens.append(self._index_to_token[idx])
            retlist.append(tokens)
        return retlist

    def merge_by_count(self, vocab: "CountVocabulary") -> None:
        if vocab.counter_dict:
            if self._counter is None:
                self._counter = collections.Counter()
            self._counter.update(vocab.counter_dict)
            self.build()

    def update(self, sentences: T.Iterable[T.Iterable[str]]) -> None:
        if self._counter is None:
            self._counter = collections.Counter()
        for tokens in sentences:
            self._counter.update(tokens)
        # Need to recreate token_to_index dictionary from counts
        self.build()

    def build(self):
        self.init()
        counts = sorted(self._counter.items(), key=operator.itemgetter(1), reverse=True)
        actual_max_vocab_size = self._max_vocab_size
        if actual_max_vocab_size is not None:
            actual_max_vocab_size += len(self._token_to_index)
        for key, count in counts:
            if actual_max_vocab_size is not None and len(self._token_to_index) >= actual_max_vocab_size:
                break
            if self._min_count and count < self._min_count:
                continue
            self.add_token(key)

    def __getstate__(self) -> T.Dict[str, T.Any]:
        """Robust pickling"""
        state = super().__getstate__()
        state["min_count"] = self._min_count
        state["max_vocab_size"] = self._max_vocab_size
        state["counter"] = self._counter
        return state

    def __setstate__(self, state) -> None:
        """Robust pickling"""
        super().__setstate__(state)
        self._min_count = state["min_count"]
        self._max_vocab_size = state["max_vocab_size"]
        self._counter = state["counter"]
        self._build_index_to_token()

    # overrides

    def add_token(self, token: str):
        self._token_to_index[token] = len(self._token_to_index)

    def init(self):
        super().init()
        self._index_to_token = []

    def init_padding(self):
        self.add_token(self._padding_token)

    def init_oov(self):
        self.add_token(self._oov_token)

    def serialize_entry(self, token: str, _: T.Any) -> str:
        return token

    def deserialize_entry(self, entry: str) -> None:
        self.add_token(entry)

    # private

    def _build_index_to_token(self):
        self._index_to_token = [None] * len(self._token_to_index)
        for token, token_id in self._token_to_index.items():
            self._index_to_token[token_id] = token
