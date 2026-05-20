"""Tests for tensorrt_model_connect.build_cli — CLI argument parsing and dispatch."""

from __future__ import annotations

import pytest

try:
    import tensorrt_model_connect.build_cli as cli
except ImportError:
    pytest.skip("tensorrt_model_connect not importable", allow_module_level=True)


class TestCliVersion:
    def test_version_returns_zero(self):
        import argparse
        ns = argparse.Namespace()
        assert cli.cmd_version(ns) == 0


class TestCliBuild:
    def test_build_missing_model_raises(self):
        """CLI should fail gracefully for nonexistent model dir."""
        import argparse
        ns = argparse.Namespace(
            model="/nonexistent/path",
            output="/tmp/test.ttrtb",
            max_cache_length=256,
            precision="fp16",
            verbose=False,
        )
        result = cli.cmd_build(ns)
        assert result == 1  # should fail, not crash


class TestCliInspect:
    def test_inspect_nonexistent(self):
        import argparse
        ns = argparse.Namespace(bundle="/nonexistent/file.ttrtb")
        result = cli.cmd_inspect(ns)
        assert result == 1

    def test_inspect_valid_bundle(self, tmp_path):
        from tensorrt_model_connect.engine_defs.torch_trt.bundle_writer import TtrtBundleInfo, write_bundle
        bundle_path = tmp_path / "test.ttrtb"
        write_bundle(str(bundle_path), TtrtBundleInfo(model_type="test"), [])

        import argparse
        ns = argparse.Namespace(bundle=str(bundle_path))
        result = cli.cmd_inspect(ns)
        assert result == 0
