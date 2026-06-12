# TrainingPeaks MCP test source manifest

This manifest exists so Fitnessbot/TrainingPeaks MCP hygiene checks can distinguish durable project tests from transient scratch files. It protects against regressions where `tests/__pycache__/test_*.pyc` remains but the source `test_*.py` file was removed by cleanup tooling.

## Required source-backed tool tests

- test_analyze.py
- test_atp_and_summary.py
- test_auth_status.py
- test_coach_composites.py
- test_coach_support.py
- test_equipment.py
- test_events.py
- test_fitness.py
- test_library.py
- test_metrics.py
- test_new_workouts.py
- test_peaks.py
- test_profile.py
- test_refresh_auth_security.py
- test_settings.py
- test_source_hygiene.py
- test_strength_references.py
- test_structure.py
- test_validation.py
- test_workout_types.py
- test_workouts.py

## Required source-backed top-level/client/auth tests

- test_server_functional.py
- test_auth/test_encrypted.py
- test_auth/test_keyring.py
- test_auth/test_validator.py
- test_client/test_http.py
- test_client/test_models.py
