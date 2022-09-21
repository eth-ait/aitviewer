from utils import FAILURE_DIR

import shutil
import pytest
import os

@pytest.fixture(scope="session", autouse=True)
def setup_failures():
    """At the start of a test session remove the failures directory"""
    if os.path.exists(FAILURE_DIR):
        shutil.rmtree(FAILURE_DIR)
