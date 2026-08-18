"""Regression tests for optional tilecpp export in tilegym.ops."""
import tilegym.ops as ops


def test_tilecpp_name_is_always_bound():
    """When the tilecpp backend is unavailable, the name must still exist."""
    assert hasattr(ops, "tilecpp")


def test_tilecpp_exported_only_when_available():
    if ops.tilecpp is None:
        assert "tilecpp" not in ops.__all__
    else:
        assert "tilecpp" in ops.__all__


def test_star_import_does_not_raise_when_tilecpp_unavailable():
    # ``from tilegym.ops import *`` previously raised AttributeError when
    # tilecpp was unavailable because ``__all__`` still referenced it.
    namespace = {}
    exec("from tilegym.ops import *", namespace)
