import io
import typing as T

DEFAULT_PADDING_TOKEN = "@@PADDING@@"
DEFAULT_OOV_TOKEN = "@@UNKNOWN@@"

S = T.TypeVar("S")  # pylint: disable=invalid-name


class Vocabulary(T.Generic[S]):
    """Manages Vocabulary. Its API is highly influenced by AllenNLP."""

    def __init__(
        self, padding: bool = True, padding_token: T.Optional[str] = None, oov_token: T.Optional[str] = None,
    ) -> None:
        self._padding = padding
        self._padding_token = padding_token or DEFAULT_PADDING_TOKEN
        self._oov_token = oov_token or DEFAULT_OOV_TOKEN
        self._token_to_index: dict = {}

    @property
    def token_to_index_dict(self) -> T.Dict[str, int]:
        return self._token_to_index

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

    def load(self, path: str, encoding="utf-8") -> None:
        """Loads a `Vocabulary` from path

        format is very simple: "<TOKEN><LINEBREAK><TOKEN><LINEBREAK>..."
        """
        with open(path, encoding=encoding) as f:
            self.loads(f)

    def loads(self, buf: T.Union[str, T.TextIO]) -> None:
        """Loads a `Vocabulary` from buffer"""
        self.init()
        for entry in buf:
            entry = entry.strip("\n")
            self.deserialize_entry(entry)

    def is_padded(self) -> bool:
        """Returns whether or not there are padding or not."""
        return self._padding

    def dump(self, path: str, encoding="utf-8") -> None:
        """dump corpus into text format."""
        with open(path, "w", encoding=encoding) as fo:
            self._dump(fo)

    def dumps(self) -> str:
        buf = io.StringIO()
        self._dump(buf)
        return buf.getvalue()

    def to_ids(self, sentences: T.Iterable[T.Iterable[str]], length: T.Optional[int] = None) -> T.List[T.List[S]]:
        """Convert from sentences into list of ids."""
        retlist = []
        for tokens in sentences:
            token_ids = []
            for token in tokens:
                token_ids.append(self._token_to_index.get(token, self._token_to_index[self._oov_token]))
            retlist.append(token_ids)
        if self._padding and length:
            padding_array = [self._token_to_index[self._padding_token]] * length
            for token_ids in retlist:
                token_ids.extend(padding_array[: length - len(token_ids)])
        return retlist

    def __getstate__(self):
        """Robust pickling"""
        return {
            "padding": self._padding,
            "padding_token": self._padding_token,
            "oov_token": self._oov_token,
            "token_to_index": self._token_to_index,
        }

    def __setstate__(self, state):
        """Robust pickling"""
        self._padding = state["padding"]
        self._padding_token = state["padding_token"]
        self._oov_token = state["oov_token"]
        self._token_to_index = state["token_to_index"]

    def __len__(self) -> int:
        return len(self._token_to_index)

    def __str__(self) -> str:
        padding_id = self.padding_id
        oov_id = self.oov_id
        vocab_size = self.vocab_size
        return f"{self.__class__.__name__}: {vocab_size} entries, padding_index={padding_id}, oov_index={oov_id}"

    def __repr__(self) -> str:
        return self.__str__()

    def init(self):
        self._token_to_index = {}
        if self._padding:
            self.init_padding()
        self.init_oov()

    def _dump(self, fobj: T.TextIO):
        need_sep = False  # To support an empty token, no linebreak at the end of file
        for token, token_id in self._token_to_index.items():
            if need_sep:
                fobj.write("\n")
            fobj.write(self.serialize_entry(token, token_id))
            need_sep = True

    def init_padding(self):
        pass

    def init_oov(self):
        pass

    def add_token(self, token: str) -> None:
        raise NotImplementedError

    def serialize_entry(self, token: str, token_id: T.Any) -> str:
        raise NotImplementedError

    def deserialize_entry(self, entry: str) -> None:
        raise NotImplementedError
