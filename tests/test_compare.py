"""Tests for the model-comparison module.

Covers record persistence (save/load roundtrip, directory creation, sort
order), comparison table formatting (column headers, CSV/MD output, N/A
handling), and the five-model guard rail.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from pathlib import Path

from pathsnap.models.base import ModelRecord
from pathsnap.evaluation.compare import (
    assert_five_models,
    build_comparison_table,
    load_records,
    save_record,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def make_record(name: str = "test_model", family: str = "test", **kwargs) -> ModelRecord:
    return ModelRecord(name=name, family=family, config={"param": 1}, seed=42, **kwargs)


# ---------------------------------------------------------------------------
# save / load
# ---------------------------------------------------------------------------

class TestSaveAndLoad:

    def test_save_and_load_roundtrip(self, tmp_path: Path):
        rec = make_record(
            cv_macro_f1=0.85,
            test_top1_accuracy=0.90,
            test_macro_f1=0.84,
            test_log_loss=0.35,
        )
        save_record(rec, tmp_path)

        loaded = load_records(tmp_path)
        assert len(loaded) == 1
        lr = loaded[0]
        assert lr.name == rec.name
        assert lr.family == rec.family
        assert lr.config == rec.config
        assert lr.seed == rec.seed
        assert lr.cv_macro_f1 == rec.cv_macro_f1
        assert lr.test_top1_accuracy == rec.test_top1_accuracy
        assert lr.test_macro_f1 == rec.test_macro_f1
        assert lr.test_log_loss == rec.test_log_loss

    def test_load_records_sorts_by_name(self, tmp_path: Path):
        save_record(make_record(name="z_model"), tmp_path)
        save_record(make_record(name="a_model"), tmp_path)

        records = load_records(tmp_path)
        names = [r.name for r in records]
        assert names == sorted(names), f"Expected sorted order, got {names}"

    def test_save_record_persists_all_contract_fields(self, tmp_path: Path):
        rec = make_record(
            name="knn",
            family="instance-based",
            cv_macro_f1=0.5,
            test_top1_accuracy=0.7,
            test_macro_f1=0.5,
            test_log_loss=None,
            route_continuity=0.9,
            inference_ms_per_obs=0.1,
            artifact_path="artifacts/p2_knn.joblib",
        )
        path = save_record(rec, tmp_path)
        data = json.loads(path.read_text(encoding="utf-8"))
        for required in [
            "name",
            "family",
            "config",
            "seed",
            "cv_macro_f1",
            "test_top1_accuracy",
            "test_macro_f1",
            "test_log_loss",
            "route_continuity",
            "inference_ms_per_obs",
            "artifact_path",
        ]:
            assert required in data, f"missing C5 field: {required}"

    def test_save_record_creates_directory(self, tmp_path: Path):
        nested = tmp_path / "sub" / "dir"
        rec = make_record(name="nested_model")
        path = save_record(rec, nested)

        assert nested.exists()
        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["name"] == "nested_model"


# ---------------------------------------------------------------------------
# comparison table
# ---------------------------------------------------------------------------

class TestBuildComparisonTable:

    @pytest.fixture()
    def five_records_dir(self, tmp_path: Path) -> Path:
        for i in range(5):
            save_record(
                make_record(
                    name=f"model_{i}",
                    family=f"family_{i}",
                    cv_macro_f1=0.80 + i * 0.01,
                    test_top1_accuracy=0.90 + i * 0.01,
                ),
                tmp_path,
            )
        return tmp_path

    def test_build_comparison_table_format(self, five_records_dir: Path):
        md = build_comparison_table(five_records_dir)

        # Must contain the standard column headers.
        for col in [
            "Model",
            "Family",
            "Key configuration",
            "CV macro F1",
            "Top-1 acc",
            "Route continuity",
            "Student-test macro F1",
            "Log loss",
            "Inference ms/obs",
        ]:
            assert col in md, f"Missing column header: {col}"

        # Must contain all 5 model names.
        for i in range(5):
            assert f"model_{i}" in md

    def test_build_comparison_table_writes_files(self, five_records_dir: Path):
        build_comparison_table(five_records_dir)

        assert (five_records_dir / "comparison.csv").exists()
        assert (five_records_dir / "comparison.md").exists()

        # CSV should have a header + 5 data rows.
        csv_lines = (five_records_dir / "comparison.csv").read_text(encoding="utf-8").strip().splitlines()
        assert len(csv_lines) == 6  # header + 5

    def test_comparison_table_handles_none_values(self, tmp_path: Path):
        """Records with None metrics should render as 'N/A', not 'None'."""
        for i in range(5):
            save_record(
                make_record(
                    name=f"model_{i}",
                    cv_macro_f1=None,
                    test_top1_accuracy=None,
                    test_log_loss=None,
                    route_continuity=None,
                    inference_ms_per_obs=None,
                ),
                tmp_path,
            )

        md = build_comparison_table(tmp_path)
        assert "None" not in md, "Table should use 'N/A', not 'None'"
        assert "N/A" in md

    def test_comparison_table_excludes_runtime_noise_from_config(self, tmp_path: Path):
        """fit_seconds is a runtime measurement, not a key configuration value."""
        for i in range(5):
            rec = make_record(name=f"m{i}")
            rec.config = {"n_neighbors": 5, "fit_seconds": 1.23}
            save_record(rec, tmp_path)

        md = build_comparison_table(tmp_path)
        assert "fit_seconds" not in md
        assert "n_neighbors=5" in md

    def test_comparison_table_uses_canonical_report_order(self, tmp_path: Path):
        """Registered ALL_MODELS names appear in report order, not alphabetical."""
        # Save in reverse order to ensure sorting is explicitly applied
        canonical = ["logreg", "knn", "random_forest", "hist_gradient_boosting", "svm_rbf"]
        for name in reversed(canonical):
            save_record(make_record(name=name, family="fam"), tmp_path)

        md = build_comparison_table(tmp_path)
        body_rows = md.strip().splitlines()[2:]  # skip header + separator
        rendered_names = [line.split("|")[1].strip() for line in body_rows]
        assert rendered_names == canonical


# ---------------------------------------------------------------------------
# five-model guard
# ---------------------------------------------------------------------------

class TestAssertFiveModels:

    def test_assert_five_models_passes(self, tmp_path: Path):
        for i in range(5):
            save_record(make_record(name=f"model_{i}"), tmp_path)

        # Should not raise.
        assert_five_models(tmp_path)

    def test_assert_five_models_fails_with_fewer(self, tmp_path: Path):
        for i in range(3):
            save_record(make_record(name=f"model_{i}"), tmp_path)

        with pytest.raises(AssertionError, match="Need at least 5 models"):
            assert_five_models(tmp_path)
