from __future__ import annotations

import sys
import unittest
from types import ModuleType
from unittest import mock

from uav_benchmark import bootstrap
from uav_benchmark.utils.gpu import resolve_gpu


class RuntimeEnvironmentTest(unittest.TestCase):
    def test_gpu_off_mode_never_probes_optional_backends(self) -> None:
        with mock.patch("builtins.__import__", side_effect=AssertionError("unexpected import")):
            info = resolve_gpu("off")

        self.assertFalse(info.enabled)
        self.assertEqual(info.backend, "numpy")
        self.assertEqual(info.device, "cpu")
        self.assertEqual(info.reason, "disabled by user")

    def test_gpu_force_mode_reports_cpu_fallback_when_backends_are_missing(self) -> None:
        real_import = __import__

        def import_without_gpu(name, *args, **kwargs):  # type: ignore[no-untyped-def]
            if name in {"cupy", "torch"}:
                raise ImportError(name)
            return real_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=import_without_gpu):
            info = resolve_gpu("force")

        self.assertFalse(info.enabled)
        self.assertEqual(info.backend, "numpy")
        self.assertEqual(info.reason, "force requested but no GPU backend found")

    def test_bootstrap_noops_when_numpy_is_healthy(self) -> None:
        original_path = list(sys.path)

        with (
            mock.patch.object(bootstrap, "_numpy_is_broken", return_value=False),
            mock.patch.object(bootstrap, "_latest_site_packages") as latest_site_packages,
        ):
            bootstrap.bootstrap_homebrew_science_stack()

        latest_site_packages.assert_not_called()
        self.assertEqual(sys.path, original_path)

    def test_bootstrap_prepends_homebrew_science_paths_and_clears_loaded_modules(self) -> None:
        original_path = list(sys.path)
        original_numpy = sys.modules.get("numpy")
        original_scipy = sys.modules.get("scipy")
        sentinel_numpy = ModuleType("numpy")
        sentinel_scipy = ModuleType("scipy")
        sys.modules["numpy"] = sentinel_numpy
        sys.modules["scipy"] = sentinel_scipy

        def fake_latest_site_packages(formula: str, relative_path: str) -> str:
            return f"/opt/homebrew/Cellar/{formula}/test/{relative_path}"

        try:
            with (
                mock.patch.object(bootstrap, "_numpy_is_broken", return_value=True),
                mock.patch.object(bootstrap, "_latest_site_packages", side_effect=fake_latest_site_packages),
            ):
                bootstrap.bootstrap_homebrew_science_stack()

            python_tag = f"python{sys.version_info.major}.{sys.version_info.minor}"
            inserted = sys.path[:3]
            self.assertIn(f"/opt/homebrew/Cellar/numpy/test/lib/{python_tag}/site-packages", inserted[0])
            self.assertIn(f"/opt/homebrew/Cellar/scipy/test/lib/{python_tag}/site-packages", inserted[1])
            self.assertIn(
                f"/opt/homebrew/Cellar/python-matplotlib/test/libexec/lib/{python_tag}/site-packages",
                inserted[2],
            )
            self.assertNotIn("numpy", sys.modules)
            self.assertNotIn("scipy", sys.modules)
        finally:
            sys.path[:] = original_path
            if original_numpy is None:
                sys.modules.pop("numpy", None)
            else:
                sys.modules["numpy"] = original_numpy
            if original_scipy is None:
                sys.modules.pop("scipy", None)
            else:
                sys.modules["scipy"] = original_scipy


if __name__ == "__main__":
    unittest.main()
