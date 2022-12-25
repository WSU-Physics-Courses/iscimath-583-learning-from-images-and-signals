"""Test Fixtures

This file is first imported by pytest and can be used to define fixtures which the tests
then use.  For example. if a test needs a temporary directory, then it can have an
argument `tmpdir`.

For more details, see

https://docs.pytest.org/fixture.html
"""
import tempfile

import pytest

import mmf_setup

# This adds the top-level folder (where `pyproject,toml` is located) to `sys.path` so
# that folders there can be imported without having to install the package.  Generally
# it is better not to do this, to ensure that only installed tests get executed.
mmf_setup.set_path()

