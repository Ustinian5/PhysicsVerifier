from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from training.openrlhf.openrlhf_contract import (
    REQUIRED_FILTER_SYMBOLS,
    REQUIRED_TRAIN_FLAGS,
    inspect_openrlhf_contract,
)


class OpenRLHFContractTests(unittest.TestCase):
    def _fake_checkout(self, root: Path, *, complete: bool) -> None:
        package = root / "openrlhf"
        train = package / "cli" / "train_ppo_ray.py"
        dynamic_filter = package / "trainer" / "ppo_utils" / "dynamic_filter.py"
        train.parent.mkdir(parents=True)
        dynamic_filter.parent.mkdir(parents=True)
        flags = REQUIRED_TRAIN_FLAGS if complete else REQUIRED_TRAIN_FLAGS[:1]
        train.write_text("\n".join(repr(flag) for flag in flags), encoding="utf-8")
        symbols = REQUIRED_FILTER_SYMBOLS if complete else REQUIRED_FILTER_SYMBOLS[:1]
        lines = []
        for name in symbols:
            if name.startswith("MODE_"):
                lines.append(f"{name} = {name!r}")
            elif name == "FilterConfig":
                lines.append("class FilterConfig:\n    pass")
            else:
                lines.append(f"def {name}():\n    pass")
        dynamic_filter.write_text("\n\n".join(lines) + "\n", encoding="utf-8")

    def test_complete_patched_contract_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._fake_checkout(root, complete=True)
            result = inspect_openrlhf_contract(root)
        self.assertTrue(result["ok"])
        self.assertEqual(result["missing_train_flags"], [])
        self.assertEqual(result["missing_filter_symbols"], [])

    def test_official_contract_without_extension_fails_early(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._fake_checkout(root, complete=False)
            result = inspect_openrlhf_contract(root)
        self.assertFalse(result["ok"])
        self.assertTrue(result["missing_train_flags"])
        self.assertTrue(result["missing_filter_symbols"])


if __name__ == "__main__":
    unittest.main()
