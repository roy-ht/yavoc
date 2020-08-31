import collections
import io
import operator
import typing as T

DEFAULT_PADDING_TOKEN = "@@PADDING@@"
DEFAULT_OOV_TOKEN = "@@UNKNOWN@@"


class Vocabulary:
    """Manages Vocabulary. Its API is highly influenced by AllenNLP."""

    def __init__(
        self,
        sentences: T.Optional[T.Iterable[T.Iterable[str]]] = None,
        padding: bool = True,
        min_count: int = 1,
        max_vocab_size: T.Optional[int] = None,
        padding_token: str = DEFAULT_PADDING_TOKEN,
        oov_token: str = DEFAULT_OOV_TOKEN,
    ) -> None:
        self._padding = padding
        self._min_count = min_count
        self._max_vocab_size = max_vocab_size
        self._padding_token = padding_token
        self._oov_token = oov_token
        self._token_to_index: dict = {}
        self._index_to_token: T.List[str] = []
        self._counter: T.Optional[collections.Counter] = None

        if sentences is not None:
            self.extend(sentences)

    @property
    def index_to_token_dict(self) -> T.Dict[int, str]:
        return self._index_to_token

    @property
    def token_to_index_dict(self) -> T.Dict[str, int]:
        return self._token_to_index

    @property
    def counter_dict(self):
        return self._counter

    @property
    def padding_id(self):
        return self._token_to_index.get(self._padding_token)

    @property
    def oov_id(self):
        return self._token_to_index.get(self._oov_token)

    @property
    def vocab_size(self):
        sz = len(self)
        if self._padding_token in self._token_to_index:
            sz -= 1
        if self._oov_token in self._token_to_index:
            sz -= 1
        return sz

    # ---- Read APIs
    def load(self, path: str, encoding="utf-8") -> None:
        """Loads a `Vocabulary` from path

        format is very simple: "<TOKEN><LINEBREAK><TOKEN><LINEBREAK>..."
        """
        self._token_to_index = {}
        self._index_to_token = []
        with open(path, encoding=encoding) as f:
            self.loads(f)

    def loads(self, buf: T.Union[str, io.TextIOBase]) -> None:
        """Loads a `Vocabulary` from buffer"""
        self._init()
        for token in buf:
            token = token.strip("\n")
            if token not in (self._padding_token, self._oov_token):
                self._token_to_index[token] = len(self._token_to_index)

    def to_ids(
        self, sentences: T.Iterable[T.Iterable[str]], padding_size: T.Optional[int] = None,
    ) -> T.List[T.List[int]]:
        """Convert from sentences into list of ids."""
        retlist = []
        for tokens in sentences:
            token_ids = []
            for token in tokens:
                token_ids.append(self._token_to_index.get(token, self._token_to_index[self._oov_token]))
            retlist.append(token_ids)
        if self._padding and padding_size:
            padding_array = [self._token_to_index[self._padding_token]] * padding_size
            for token_ids in retlist:
                token_ids.extend(padding_array[: padding_size - len(token_ids)])
        return retlist

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

    def is_padded(self) -> bool:
        """Returns whether or not there are padding or not."""
        return self._padding

    # ---- Write APIs
    def merge_by_count(self, vocab: "Vocabulary") -> None:
        cd = vocab.counter_dict
        if cd:
            if self._counter is None:
                self._counter = collections.Counter()
            self._counter.update(cd)
            self._rebuild()

    def extend(self, sentences: T.Iterable[T.Iterable[str]]) -> None:
        if self._counter is None:
            self._counter = collections.Counter()
        for tokens in sentences:
            self._counter.update(tokens)
        # Need to recreate token_to_index dictionary from counts
        self._rebuild()

    def change_threshold(self, min_count: T.Optional[int] = None, max_vocab_size: T.Optional[int] = None) -> None:
        need_rebuild = False
        if min_count is not None and min_count != self._min_count:
            self._min_count = min_count
            need_rebuild = True
        if max_vocab_size is not None and max_vocab_size != self._max_vocab_size:
            self._max_vocab_size = max_vocab_size
            need_rebuild = True
        if need_rebuild:
            self._rebuild()

    def dump(self, path, encoding="utf-8") -> None:
        """dump corpus into text format."""
        with open(path, "w", encoding=encoding) as fo:
            self._dump(fo)

    def dumps(self) -> str:
        buf = io.StringIO()
        self._dump(buf)
        return buf.getvalue()

    def __getstate__(self):
        """Robust pickling"""
        return {
            "padding": self._padding,
            "min_count": self._min_count,
            "max_vocab_size": self._max_vocab_size,
            "padding_token": self._padding_token,
            "oov_token": self._oov_token,
            "token_to_index": self._token_to_index,
            "counter": self._counter,
        }

    def __setstate__(self, state):
        """Robust pickling"""
        self._padding = state["padding"]
        self._min_count = state["min_count"]
        self._max_vocab_size = state["max_vocab_size"]
        self._padding_token = state["padding_token"]
        self._oov_token = state["oov_token"]
        self._token_to_index = state["token_to_index"]
        self._build_index_to_token()
        self._counter = state["counter"]

    def __len__(self) -> int:
        return len(self._token_to_index)

    def __str__(self) -> str:
        padding_id = self.padding_id
        oov_id = self.oov_id
        vocab_size = self.vocab_size
        return f"Vocabulary: {vocab_size} entries, padding_index={padding_id}, oov_index={oov_id}"

    def __repr__(self) -> str:
        return self.__str__()

    def _init(self):
        self._token_to_index = {}
        self._index_to_token = {}
        if self._padding:
            self._token_to_index[self._padding_token] = len(self._token_to_index)
        self._token_to_index[self._oov_token] = len(self._token_to_index)

    def _rebuild(self):
        self._init()
        counts = sorted(self._counter.items(), key=operator.itemgetter(1), reverse=True)
        actual_max_vocab_size = self._max_vocab_size
        if actual_max_vocab_size is not None:
            actual_max_vocab_size += len(self._token_to_index)
        for key, count in counts:
            if len(self._token_to_index) >= actual_max_vocab_size:
                break
            if self._min_count and count < self._min_count:
                continue
            self._token_to_index[key] = len(self._token_to_index)

    def _build_index_to_token(self):
        self._index_to_token = [None] * len(self._token_to_index)
        for token, token_id in self._token_to_index.items():
            self._index_to_token[token_id] = token

    def _dump(self, fobj):
        need_sep = False  # To support an empty token, no linebreak at the end of file
        for token in self._token_to_index:
            if need_sep:
                fobj.write("\n")
            fobj.write(token)
            need_sep = True
