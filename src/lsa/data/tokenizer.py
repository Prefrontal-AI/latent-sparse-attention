"""Resolve and load the tokenizer named by a dataset card."""

from typing import TYPE_CHECKING, Any

from lsa import env

if TYPE_CHECKING:
    from lsa import assets

# Before a caller reads a tokenizer or a hub dataset - the hub reads its offline switches at import
env.prefer_offline_hub()


def tokenizer_repo(card: "assets.BenchmarkCard | assets.TokenizedCard") -> str:
    """Return the hub id of the tokenizer a dataset's lengths are counted in, as its model card names it.

    The asset loader is imported on use rather than at import, because it reaches omegaconf and a reader of
    the corpus that never tokenizes should not.

    Args:
        card: The benchmark or tokenized dataset whose `tokenizer` names a model card.
    """
    from lsa import assets

    return assets.model(card.tokenizer).repo


def tokenizer_checkpoint(card: "assets.BenchmarkCard | assets.TokenizedCard") -> str:
    """Return the pinned snapshot containing a dataset's tokenizer."""
    from lsa import assets

    return assets.model(card.tokenizer).checkpoint


def load_tokenizer(path: str, *, require_fast: bool = False, trust_remote_code: bool = False) -> Any:
    """Load a tokenizer, optionally requiring the fast implementation.

    Args:
        path: A hub id or a pinned snapshot directory.
        require_fast: Refuse a slow tokenizer when preparing a large corpus.
        trust_remote_code: Allow the checkpoint's tokenizer implementation.

    Returns:
        The tokenizer.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=trust_remote_code)
    if require_fast and not tokenizer.is_fast:
        raise ValueError(f"{path} has no fast tokenizer, and tokenizing a corpus needs one")
    return tokenizer
