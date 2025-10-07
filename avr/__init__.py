"""test package initializer.

This file makes the `test` directory a proper Python package so intra-repo
imports like `from test.test_media_utils import ...` resolve when the
repository root is on sys.path.

Keep this file minimal to avoid side-effects when importing the package.
"""

__all__ = []
