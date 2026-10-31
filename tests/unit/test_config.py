"""A checkpoint's config is read through to its decoder."""

from types import SimpleNamespace

from lsa.config import text_config


def test_a_multimodal_config_is_read_through_to_its_decoder():
    decoder = SimpleNamespace(head_dim=128)
    assert text_config(SimpleNamespace(text_config=decoder)) is decoder


def test_a_published_config_is_read_through_to_its_decoder():
    decoder = {"vocab_size": 8}
    assert text_config({"text_config": decoder}) is decoder


def test_a_text_only_config_is_its_own_decoder():
    published = {"vocab_size": 8}
    assert text_config(published) is published
    bare = SimpleNamespace(head_dim=128)
    assert text_config(bare) is bare
