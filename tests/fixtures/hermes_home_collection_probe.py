"""Import-time probe used by the pre-collection HERMES_HOME regression test."""

import os
from pathlib import Path


output = os.environ.get("HERMES_COLLECTION_PROBE_OUTPUT")
if output:
    Path(output).write_text(os.environ.get("HERMES_HOME", ""), encoding="utf-8")


def test_probe_collected():
    pass