"""Tests for the tilegym.ops package init."""


def test_tilecpp_not_exported_when_unavailable():
    """If tilecpp is not imported, it must not appear in __all__."""
    import tilegym.ops as ops

    if not hasattr(ops, "tilecpp") or ops.tilecpp is None:
        assert "tilecpp" not in ops.__all__


