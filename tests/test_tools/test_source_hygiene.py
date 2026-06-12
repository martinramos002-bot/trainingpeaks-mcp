"""Guard durable TrainingPeaks MCP test sources from pycache-only regressions."""

from __future__ import annotations

from pathlib import Path

EXPECTED_TEST_TOOL_SOURCES = {
    "test_analyze.py",
    "test_atp_and_summary.py",
    "test_auth_status.py",
    "test_coach_composites.py",
    "test_coach_support.py",
    "test_equipment.py",
    "test_events.py",
    "test_fitness.py",
    "test_library.py",
    "test_metrics.py",
    "test_new_workouts.py",
    "test_peaks.py",
    "test_profile.py",
    "test_refresh_auth_security.py",
    "test_settings.py",
    "test_source_hygiene.py",
    "test_strength_references.py",
    "test_structure.py",
    "test_validation.py",
    "test_workout_types.py",
    "test_workouts.py",
}


def test_expected_tool_test_sources_are_present():
    tests_dir = Path(__file__).resolve().parent
    manifest = tests_dir.parent / "TEST_SOURCE_MANIFEST.md"
    present = {path.name for path in tests_dir.glob("test_*.py")}

    assert manifest.exists()
    assert present >= EXPECTED_TEST_TOOL_SOURCES


def test_no_test_tool_pycache_without_source():
    tests_dir = Path(__file__).resolve().parent
    pycache = tests_dir / "__pycache__"
    if not pycache.exists():
        return

    missing_sources = []
    for pyc in pycache.glob("test_*.pyc"):
        source_name = pyc.name.split(".cpython-", 1)[0] + ".py"
        if not (tests_dir / source_name).exists():
            missing_sources.append(source_name)

    assert missing_sources == []
