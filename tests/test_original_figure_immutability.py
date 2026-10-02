from __future__ import annotations

import hashlib
from pathlib import Path


def test_original_paper_figures_match_protected_checksums():
    root = Path(__file__).resolve().parents[1]
    checksum_file = root / "schemas" / "protected_figures.sha256"
    for line in checksum_file.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split(None, 1)
        path = root / relative
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == expected, f"Protected paper figure changed: {relative}"
