"""Tests for Fitnessbot coach composite helpers.

These are pure helpers first: no TrainingPeaks network calls, no writes.
"""

from datetime import date, timedelta

import pytest

import tp_mcp.tools.coach_composites as coach_composites
from tp_mcp.coach_composite_helpers import (
    classify_missed_workout_reason,
    classify_readiness_snapshot,
    normalize_subjective_feedback,
    summarize_feedback_patterns,
    validate_week_plan_guardrails,
    week_bounds,
)


def test_week_bounds_returns_monday_to_sunday_for_midweek_date():
    start, end = week_bounds(date(2026, 5, 21))

    assert start.isoformat() == "2026-05-18"
    assert end.isoformat() == "2026-05-24"


def test_classify_missed_workout_reason_uses_athlete_comment_time_not_recovery_guess():
    result = classify_missed_workout_reason(
        {
            "title": "Run recuperación/técnica",
            "completed": None,
            "workout_comments": [
                {"comment": "No tuve tiempo para hacer el entrenamiento", "isCoach": False}
            ],
            "new_comment": None,
        },
        private_note={"note": ""},
        calendar_notes=[{"title": "Disponibilidad", "description": "Lunes 1h30"}],
    )

    assert result["category"] == "time_logistics"
    assert result["confidence"] == "high"
    assert "No tuve tiempo" in result["evidence"]
    assert result["decision_rule"] == "do_not_compensate_missed_tss"


def test_classify_readiness_snapshot_flags_orange_when_sleep_hrv_and_tsb_are_bad():
    result = classify_readiness_snapshot(
        latest_metrics={"sleep_hours": 5.2, "hrv": 45, "pulse": 58},
        baselines={"hrv": 65, "pulse": 48},
        fitness={"tsb": -22, "atl": 45, "ctl": 20},
        subjective_flags=["fatigue"],
    )

    assert result["traffic_light"] == "orange"
    assert "sleep_low" in result["flags"]
    assert "hrv_low_vs_baseline" in result["flags"]
    assert "tsb_very_negative" in result["flags"]
    assert result["decision_bias"] == "reduce_or_recovery"


def test_classify_readiness_snapshot_green_when_signals_are_normal():
    result = classify_readiness_snapshot(
        latest_metrics={"sleep_hours": 8.0, "hrv": 63, "pulse": 49},
        baselines={"hrv": 65, "pulse": 48},
        fitness={"tsb": -4, "atl": 26, "ctl": 24},
        subjective_flags=[],
    )

    assert result["traffic_light"] == "green"
    assert result["flags"] == []
    assert result["decision_bias"] == "execute_as_planned"


def test_metric_dict_for_date_accepts_trainingpeaks_timestamp_camel_case():
    latest, freshness = coach_composites._metric_dict_for_date(
        {
            "metrics": [
                {"timeStamp": "2026-06-10T00:00:00", "details": [{"label": "HRV", "value": 72}]},
            ]
        },
        "2026-06-10",
    )

    assert latest["timeStamp"] == "2026-06-10T00:00:00"
    assert latest["hrv"] == 72
    assert freshness["status"] == "current"
    assert freshness["source_date"] == "2026-06-10"
    assert freshness["is_current_day"] is True


def test_body_battery_interpretation_reports_trainingpeaks_array_with_wake_value():
    result = coach_composites._body_battery_interpretation({"body_battery": [63, 100, 90.4085]})

    assert result is not None
    assert result["format"] == "array_min_max_avg"
    assert result["min"] == 63
    assert result["max"] == 100
    assert result["avg"] == 90.4
    assert result["wake_value"] == 100
    assert result["recovery_support"] == "strong"
    assert "100 al despertar" in result["display_guidance"]
    assert "no permiso para sumar carga" in result["display_guidance"]
    assert "wake/end-of-sleep value by Garmin" in result["coaching_rule"]


def test_metric_dict_for_date_marks_yesterday_fallback_as_stale():
    latest, freshness = coach_composites._metric_dict_for_date(
        {
            "metrics": [
                {"date": "2026-06-01", "sleep_hours": 7.5, "hrv": 64},
                {"date": "2026-06-02", "sleep_hours": 7.55, "hrv": 71},
            ]
        },
        "2026-06-03",
    )

    assert latest["date"] == "2026-06-02"
    assert freshness["status"] == "stale"
    assert freshness["source_date"] == "2026-06-02"
    assert freshness["is_current_day"] is False
    assert freshness["decision_rule"] == "do_not_use_stale_readiness_as_today"


@pytest.mark.asyncio
async def test_readiness_snapshot_never_presents_stale_metrics_as_today(monkeypatch):
    async def fake_metrics(start_date, end_date):
        return {
            "metrics": [
                {"date": "2026-06-01", "sleep_hours": 7.0, "hrv": 60, "pulse": 47},
                {"date": "2026-06-02", "sleep_hours": 7.55, "hrv": 71, "pulse": 46},
            ],
            "date_range": {"start": start_date, "end": end_date},
        }

    async def fake_fitness(**_kwargs):
        return {"daily_data": [{"date": "2026-06-03", "tsb": -14.6, "atl": 40, "ctl": 25}]}

    monkeypatch.setattr(coach_composites, "tp_get_metrics", fake_metrics)
    monkeypatch.setattr(coach_composites, "tp_get_fitness", fake_fitness)

    result = await coach_composites.tp_coach_readiness_snapshot("2026-06-03")

    assert result["date"] == "2026-06-03"
    assert result["latest_metrics"]["date"] == "2026-06-02"
    assert result["metric_freshness"]["status"] == "stale"
    assert result["decision_guardrails"]["readiness_metrics_are_current_day"] is False
    assert result["decision_guardrails"]["must_not_use_stale_readiness_as_today"] is True
    assert result["decision_guardrails"]["explain_freshness_status_plainly"] is True
    assert "Body Battery" in result["decision_guardrails"]["affected_readiness_signals_to_name"]
    assert "source_date/requested_date" in result["decision_guardrails"]["if_stale_or_missing"]
    assert "wake/end-of-sleep value by Garmin" in result["decision_guardrails"]["body_battery_rule"]
    assert "highest and lowest" in result["decision_guardrails"]["body_battery_rule"]
    assert "not an alarm" in result["decision_guardrails"]["load_language_rule"]
    assert "mechanical legs/impact" in result["decision_guardrails"]["combined_readiness_load_rule"]
    assert result["classification"]["traffic_light"] == "unknown_stale_metrics"
    assert result["classification"]["decision_bias"] == "conservative_until_same_day_metrics_arrive"
    assert "readiness_metrics_not_current_day" in result["classification"]["flags"]
    assert result["readiness_layers"]["session_decision"]["traffic_light"] == "unknown_stale_metrics"


@pytest.mark.asyncio
async def test_readiness_snapshot_separates_physiology_from_session_load_and_recovery_rules(monkeypatch):
    async def fake_metrics(start_date, end_date):
        return {
            "metrics": [
                {
                    "timeStamp": "2026-06-10T00:00:00",
                    "details": [
                        {"label": "HRV", "value": 72},
                        {"label": "Sleep Hours", "value": 8.2167},
                        {"label": "Pulse", "value": 46},
                        {"label": "Body Battery", "value": [63, 100, 90.4085]},
                    ],
                }
            ],
            "date_range": {"start": start_date, "end": end_date},
        }

    async def fake_fitness(**_kwargs):
        return {"daily_data": [{"date": "2026-06-10", "tsb": -10.7, "atl": 40, "ctl": 29}]}

    monkeypatch.setattr(coach_composites, "tp_get_metrics", fake_metrics)
    monkeypatch.setattr(coach_composites, "tp_get_fitness", fake_fitness)

    result = await coach_composites.tp_coach_readiness_snapshot("2026-06-10")

    assert result["metric_freshness"]["is_current_day"] is True
    assert result["readiness_layers"]["physiological_readiness"]["traffic_light"] == "green"
    assert result["readiness_layers"]["session_decision"]["traffic_light"] == "yellow"
    assert "physiological readiness is green" in result["readiness_layers"]["reporting_rule"]
    assert result["body_battery_interpretation"]["display_guidance"].startswith("Body Battery:")
    assert "al despertar" in result["body_battery_interpretation"]["display_guidance"]
    assert result["body_battery_interpretation"]["wake_value"] == 100
    assert "does not authorize extending" in result["decision_guardrails"]["good_readiness_does_not_add_load_rule"]
    assert "10-15 W" in result["decision_guardrails"]["recovery_execution_fallback_rule"]
    assert "3-5 evidence bullets" in result["decision_guardrails"]["brevity_rule"]


def test_validate_week_plan_guardrails_rejects_unavailable_day_and_multiple_priorities():
    plan = {
        "priority": ["long_run", "threshold"],
        "days": [
            {"date": "2026-05-19", "sport": "Run", "intensity": "easy", "duration_min": 45},
            {"date": "2026-05-20", "sport": "Run", "intensity": "threshold", "duration_min": 60},
        ],
    }
    availability = {"2026-05-20": {"available": False}}

    result = validate_week_plan_guardrails(plan, availability)

    assert result["ok"] is False
    assert "multiple_primary_priorities" in result["violations"]
    assert "training_on_unavailable_day:2026-05-20" in result["violations"]


def test_validate_week_plan_guardrails_warns_when_strength_missing():
    plan = {
        "priority": "z2_frequency",
        "days": [
            {"date": "2026-05-19", "sport": "Run", "intensity": "easy", "duration_min": 45},
            {"date": "2026-05-21", "sport": "Bike", "intensity": "z2", "duration_min": 60},
        ],
    }

    result = validate_week_plan_guardrails(plan, {})

    assert result["ok"] is True
    assert "strength_or_mobility_missing" in result["warnings"]
    # Without long-run history the 10% rule emits a soft warning, not a violation.
    assert "long_run_10pct_rule_not_evaluated_no_history" in result["warnings"]
    assert result["summary"]["long_run_10pct_rule"]["active"] is False


def _recent_iso(days_ago: int) -> str:
    return (date.today() - timedelta(days=days_ago)).isoformat()


def test_validate_week_plan_guardrails_blocks_long_run_exceeding_10pct_rule():
    """A planned long run above 110% of the 30-day max is a violation."""
    plan = {
        "priority": "long_run",
        "days": [
            # 18 km long run when the recent max is 15 km → cap is 16.5 km.
            {"date": "2026-06-21", "sport": "Run", "intensity": "long", "duration_min": 110, "distance_km": 18.0},
            {"date": "2026-06-22", "sport": "Run", "intensity": "easy", "duration_min": 40},
            {"date": "2026-06-23", "sport": "Strength", "intensity": "fuerza", "duration_min": 30},
        ],
    }
    history = [
        {"date": _recent_iso(10), "distance_km": 15.0},
        {"date": _recent_iso(20), "distance_km": 12.0},
    ]

    result = validate_week_plan_guardrails(plan, {}, long_run_history=history)

    assert result["ok"] is False
    long_run_violations = [v for v in result["violations"] if v.startswith("long_run_exceeds_10pct_rule")]
    assert len(long_run_violations) == 1
    assert "2026-06-21" in long_run_violations[0]
    assert "18.0km" in long_run_violations[0]
    # Cap = 15 * 1.10 = 16.5
    assert "16.5km" in long_run_violations[0]
    assert result["summary"]["long_run_10pct_rule"]["active"] is True
    assert result["summary"]["long_run_10pct_rule"]["recent_long_run_max_km"] == 15.0
    assert result["summary"]["long_run_10pct_rule"]["cap_km"] == 16.5


def test_validate_week_plan_guardrails_allows_long_run_within_10pct_rule():
    """A planned long run at exactly 110% of the max is allowed (boundary)."""
    plan = {
        "priority": "long_run",
        "days": [
            # 16.5 km == cap (15 * 1.10); not strictly greater, so allowed.
            {"date": "2026-06-21", "sport": "Run", "intensity": "long", "duration_min": 110, "distance_km": 16.5},
            {"date": "2026-06-22", "sport": "Run", "intensity": "easy", "duration_min": 40},
            {"date": "2026-06-23", "sport": "Strength", "intensity": "fuerza", "duration_min": 30},
        ],
    }
    history = [{"date": _recent_iso(7), "distance_km": 15.0}]

    result = validate_week_plan_guardrails(plan, {}, long_run_history=history)

    assert result["ok"] is True
    assert not any(v.startswith("long_run_exceeds_10pct_rule") for v in result["violations"])
    assert result["summary"]["long_run_10pct_rule"]["active"] is True


def test_validate_week_plan_guardrails_ignores_long_run_history_older_than_30_days():
    """Entries older than 30 days must not count toward the recent max."""
    plan = {
        "priority": "long_run",
        "days": [
            # 14 km exceeds the 13.2 km cap derived from the recent 12 km max,
            # but would pass if the 60-day-old 30 km entry were (wrongly)
            # counted (cap would be 33 km). This proves only recent entries drive
            # the rule.
            {"date": "2026-06-21", "sport": "Run", "intensity": "long", "duration_min": 110, "distance_km": 14.0},
            {"date": "2026-06-22", "sport": "Strength", "intensity": "fuerza", "duration_min": 30},
        ],
    }
    # 60-day-old 30 km run should be ignored; only the 10-day-old 12 km counts.
    # Cap from recent = 13.2; 14 > 13.2 → violation.
    history = [
        {"date": _recent_iso(60), "distance_km": 30.0},
        {"date": _recent_iso(10), "distance_km": 12.0},
    ]

    result = validate_week_plan_guardrails(plan, {}, long_run_history=history)

    # The 14 km run violates the 13.2 km cap derived from the recent 12 km max.
    assert result["ok"] is False
    assert result["summary"]["long_run_10pct_rule"]["recent_long_run_max_km"] == 12.0
    assert result["summary"]["long_run_10pct_rule"]["cap_km"] == round(12.0 * 1.10, 2)
    long_run_violations = [v for v in result["violations"] if v.startswith("long_run_exceeds_10pct_rule")]
    assert len(long_run_violations) == 1


def test_validate_week_plan_guardrails_10pct_rule_only_checks_run_sessions():
    """Bike/strength sessions with large distances must not trigger the run rule."""
    plan = {
        "priority": "z2_base",
        "days": [
            # A 40 km bike ride must not be flagged by the run-only 10% rule.
            {"date": "2026-06-21", "sport": "Bike", "intensity": "z2", "duration_min": 120, "distance_km": 40.0},
            {"date": "2026-06-22", "sport": "Run", "intensity": "easy", "duration_min": 40, "distance_km": 6.0},
            {"date": "2026-06-23", "sport": "Strength", "intensity": "fuerza", "duration_min": 30},
        ],
    }
    history = [{"date": _recent_iso(7), "distance_km": 15.0}]

    result = validate_week_plan_guardrails(plan, {}, long_run_history=history)

    assert result["ok"] is True
    assert not any(v.startswith("long_run_exceeds_10pct_rule") for v in result["violations"])


def test_summarize_feedback_patterns_maps_feeling_code_and_counts_risk_flags():
    workouts = [
        {"rpe": 2, "feeling": 5, "workout_comments": "Calor fuerte, sin dolor"},
        {"rpe": 7, "feeling": 7, "workout_comments": "DOMS gemelos y dolor leve"},
        {"rpe": 4, "feeling": 3, "workout_comments": "Buen control"},
    ]

    result = summarize_feedback_patterns(workouts)

    assert result["count"] == 3
    assert result["feeling_labels"] == ["Normal", "Débil", "Fuerte"]
    assert result["flags"]["heat"] == 1
    assert result["flags"]["pain"] == 1
    assert result["flags"]["doms"] == 1


def test_normalize_subjective_feedback_maps_code_7_as_debil_not_positive():
    result = normalize_subjective_feedback({"rpe": 2, "feeling": 7})

    assert result["rpe"] == 2.0
    assert result["feeling"]["code"] == 7
    assert result["feeling"]["label"] == "Débil"
    assert result["feeling"]["score_1_to_5"] == 2
    assert result["feeling"]["display"] == "Feeling Débil (2/5; TP code 7)"
    assert result["feeling"]["warning"] == "feeling_scale_is_inverse_higher_is_worse"


def test_science_guardrails_flag_heat_reds_uncertainty_without_diagnosis():
    result = coach_composites._build_science_guardrails(
        workout_summary={
            "rpe7_summary": {"total_count": 2},
            "session_frequency": {"sessions_per_week": 4.0},
            "weekly_tss": [{"tss": 120}, {"tss": 180}, {"tss": 260}],
        },
        metrics_compact={
            "available": True,
            "latest": {"sleep_hours": 5.4, "stress": 42},
            "averages": {"sleep_hours": 7.0},
        },
        fitness_compact={"current": {"ctl": 30, "atl": 45, "tsb": -15}},
        fitness_historical=None,
        expanded_workouts=[
            {"title": "Run largo", "detail": {"workout_comments": "Calor fuerte, fatiga y gel insuficiente"}},
            {"title": "Run easy", "detail": {"workout_comments": "Fatiga residual, sin dolor"}},
        ],
        period_days=14,
        cycle_status={"test_cp_ftp_scheduled": "Jul 20-26, 2026"},
        pain_mentions=0,
        pw_hr_status={"available_in_period_review_context": False},
        consistency_warnings=["example_conflict"],
    )

    assert result["heat_environment"]["mentions"] == 1
    assert result["heat_environment"]["fueling_or_hydration_mentions"] == 1
    assert "repeated_fatigue_language" in result["reds_energy_availability"]["flags"]
    assert "sleep_low_energy_availability_context_needed" in result["reds_energy_availability"]["flags"]
    assert result["cp_quality"]["next_test_window"] == "Jul 20-26, 2026"
    assert result["load_distribution"]["rpe7_count"] == 2
    assert result["uncertainty"]["confidence"] == "low_until_reconciled"
    assert "avoid ACWR-style causal injury claims" in result["load_distribution"]["rule"]
    assert "not diagnoses" in result["pain_safety"]["rule"]


@pytest.mark.asyncio
async def test_daily_brief_context_expands_missed_yesterday_detail_and_classifies_reason(monkeypatch):
    calls: list[tuple[str, str | None]] = []

    async def fake_readiness(**_kwargs):
        return {"classification": {"traffic_light": "yellow"}}

    async def fake_week_context(week_of=None):
        return {
            "week": {"start": "2026-06-01", "end": "2026-06-07"},
            "weekly_summary": {
                "workouts": [
                    {"date": "2026-06-01", "type": "completed", "tss_planned": 40, "tss_actual": 42},
                    {"date": "2026-06-02", "type": "planned", "tss_planned": 30, "tss_actual": None},
                ]
            },
            "workouts": {},
        }

    async def fake_workouts(start_date, end_date, workout_filter="all"):
        calls.append(("workouts", f"{start_date}:{end_date}:{workout_filter}"))
        return {
            "workouts": [
                {
                    "id": "missed-1",
                    "date": "2026-05-31",
                    "title": "Run recuperación/técnica",
                    "type": "planned",
                    "duration_actual": None,
                    "tss_actual": None,
                    "tss_planned": 20,
                },
                {"id": "today-1", "date": "2026-06-01", "title": "Fuerza base + movilidad", "type": "planned"},
            ],
        }

    async def fake_detail(workout_id):
        calls.append(("detail", workout_id))
        return {
            "id": workout_id,
            "date": "2026-05-31",
            "title": "Run recuperación/técnica",
            "completed": None,
            "workout_comments": [
                {"comment": "No tuve tiempo para hacer el entrenamiento", "isCoach": False}
            ],
            "metrics": {"duration_actual": None, "tss_actual": None, "tss_planned": 20},
        }

    async def fake_note(workout_id):
        calls.append(("private_note", workout_id))
        return {"workout_id": workout_id, "note": "", "updated_at": None}

    async def fake_calendar_notes(start_date, end_date):
        calls.append(("calendar_notes", f"{start_date}:{end_date}"))
        return {"notes": []}

    monkeypatch.setattr(coach_composites, "tp_coach_readiness_snapshot", fake_readiness)
    monkeypatch.setattr(coach_composites, "tp_coach_week_context", fake_week_context)
    monkeypatch.setattr(coach_composites, "tp_get_workouts", fake_workouts)
    monkeypatch.setattr(coach_composites, "tp_get_workout", fake_detail)
    monkeypatch.setattr(coach_composites, "tp_get_workout_note", fake_note)
    monkeypatch.setattr(coach_composites, "tp_list_notes", fake_calendar_notes)

    result = await coach_composites.tp_coach_daily_brief_context("2026-06-01")

    yesterday = result["yesterday"][0]
    assert ("detail", "missed-1") in calls
    assert ("private_note", "missed-1") in calls
    assert yesterday["status"] == "missed_or_uncompleted"
    assert yesterday["reason"]["category"] == "time_logistics"
    assert "No tuve tiempo" in yesterday["reason"]["evidence"]
    assert result["decision_guardrails"]["can_interpret_missed_yesterday"] is True
    assert "tp_get_workout_detail_and_private_note_called_for_missed_yesterday" in result["verification"]
    contract = result["daily_brief_output_contract"]
    assert "Readiness fisiológica" in contract["headline_rule"]
    assert contract["closure_required_when_space_allows"] == [
        "Confianza",
        "Qué voy a vigilar",
        "Dato faltante",
        "Qué cambiaría la decisión",
    ]
    tss = result["tss_language_contract"]
    assert tss["completed_actual_tss"] == 42.0
    assert tss["planned_original_tss"] == 70.0
    assert tss["projected_tss_if_remaining_completed"] == 72.0
    assert "Never call" in tss["wording_rule"]


def test_period_expansion_does_not_classify_future_planned_workout_as_missed():
    workout = {
        "id": "future-1",
        "date": "2099-01-01",
        "title": "Run recuperación/técnica",
        "type": "planned",
        "tss_planned": 20,
        "tss_actual": None,
    }

    assert "planned_not_completed_or_missing_actuals" not in coach_composites._period_expansion_reasons(workout)


@pytest.mark.asyncio
async def test_period_review_context_expands_missed_key_and_anomalous_workouts(monkeypatch):
    calls: list[tuple[str, str | None]] = []

    async def fake_workouts(start_date, end_date, workout_filter="all"):
        calls.append(("workouts", f"{start_date}:{end_date}:{workout_filter}"))
        return {
            "workouts": [
                {
                    "id": "missed-1",
                    "date": "2026-05-31",
                    "title": "Run recuperación/técnica",
                    "type": "planned",
                    "tss_planned": 20,
                    "tss_actual": None,
                },
                {
                    "id": "key-1",
                    "date": "2026-06-02",
                    "title": "Run largo controlado",
                    "type": "completed",
                    "duration_actual": 76,
                    "tss_actual": 53,
                },
                {
                    "id": "easy-1",
                    "date": "2026-06-04",
                    "title": "Bike Z2 suave",
                    "type": "completed",
                    "duration_actual": 45,
                    "tss_actual": 22,
                },
            ],
        }

    async def fake_detail(workout_id):
        calls.append(("detail", workout_id))
        if workout_id == "missed-1":
            return {
                "id": workout_id,
                "title": "Run recuperación/técnica",
                "completed": None,
                "workout_comments": [{"comment": "No tuve tiempo para hacer el entrenamiento"}],
                "metrics": {"tss_planned": 20, "tss_actual": None},
            }
        return {
            "id": workout_id,
            "title": "Run largo controlado",
            "completed": True,
            "rpe": 5,
            "feeling": 5,
            "workout_comments": "Buen control, sin dolor",
            "metrics": {"duration_actual": 76, "tss_actual": 53},
        }

    async def fake_note(workout_id):
        calls.append(("private_note", workout_id))
        return {"workout_id": workout_id, "note": ""}

    async def fake_generic(*_args, **_kwargs):
        return {"notes": [], "availability": [], "metrics": [], "daily_data": [], "ftp_bike": 206}

    monkeypatch.setattr(coach_composites, "tp_get_workouts", fake_workouts)
    monkeypatch.setattr(coach_composites, "tp_get_workout", fake_detail)
    monkeypatch.setattr(coach_composites, "tp_get_workout_note", fake_note)
    monkeypatch.setattr(coach_composites, "tp_get_fitness", fake_generic)
    monkeypatch.setattr(coach_composites, "tp_get_metrics", fake_generic)
    monkeypatch.setattr(coach_composites, "tp_get_athlete_settings_summary", fake_generic)
    monkeypatch.setattr(coach_composites, "tp_list_notes", fake_generic)
    monkeypatch.setattr(coach_composites, "tp_get_availability", fake_generic)

    result = await coach_composites.tp_coach_period_review_context("2026-05-31", "2026-06-06", "weekly")

    expanded_ids = {item["id"] for item in result["expanded_workout_highlights"]}
    assert expanded_ids == {"missed-1", "key-1"}
    assert ("detail", "missed-1") in calls
    assert ("private_note", "missed-1") in calls
    assert ("detail", "key-1") in calls
    assert ("detail", "easy-1") not in calls
    missed = next(item for item in result["expanded_workout_highlights"] if item["id"] == "missed-1")
    assert missed["reason"]["category"] == "time_logistics"
    assert result["decision_guardrails"]["must_expand_notable_workouts_before_causal_claims"] is True
    assert result["decision_guardrails"]["can_make_period_causal_claims"] is True


@pytest.mark.asyncio
async def test_week_context_advertises_daily_brief_protocol(monkeypatch):
    async def fake_weekly_summary(**_kwargs):
        return {"planned_tss": 120, "actual_tss": 90}

    async def fake_workouts(*_args, **_kwargs):
        return []

    async def fake_fitness(*_args, **_kwargs):
        return {"daily_data": []}

    async def fake_settings(*_args, **_kwargs):
        return {"ftp_bike": 206}

    async def fake_metrics(*_args, **_kwargs):
        return {"metrics": []}

    async def fake_notes(*_args, **_kwargs):
        return {"notes": []}

    async def fake_availability(*_args, **_kwargs):
        return {"availability": []}

    monkeypatch.setattr(coach_composites, "tp_get_weekly_summary", fake_weekly_summary)
    monkeypatch.setattr(coach_composites, "tp_get_workouts", fake_workouts)
    monkeypatch.setattr(coach_composites, "tp_get_fitness", fake_fitness)
    monkeypatch.setattr(coach_composites, "tp_get_athlete_settings_summary", fake_settings)
    monkeypatch.setattr(coach_composites, "tp_get_metrics", fake_metrics)
    monkeypatch.setattr(coach_composites, "tp_list_notes", fake_notes)
    monkeypatch.setattr(coach_composites, "tp_get_availability", fake_availability)

    result = await coach_composites.tp_coach_week_context("2026-05-28")

    protocol = result["daily_brief_protocol"]
    required = protocol["required_context"]
    assert "yesterday_planned_vs_completed_with_workout_comment_private_note_and_calendar_notes" in required
    assert "week_to_date_planned_vs_completed_sessions_and_tss" in required
    assert "remaining_week_key_sessions_and_availability" in required
    assert "readiness_metric_freshness_with_affected_signals" in required
    assert "do not compensate missed TSS automatically" in protocol["decision_rule"]
    freshness_contract = protocol["readiness_freshness_contract"]
    assert "absent, stale, partial" in freshness_contract["if_stale_partial_or_missing"]
    assert "wake/end-of-sleep" in freshness_contract["body_battery_rule"]
    assert "rango TP 63→100" in freshness_contract["body_battery_rule"]
    assert "not an alarm" in freshness_contract["load_language_rule"]
    assert "legs/impact" in freshness_contract["combined_readiness_load_rule"]
    assert "Readiness fisiológica: verde" in freshness_contract["two_light_rule"]
    assert "🟢 Hoy" in freshness_contract["two_light_rule"]
    assert "executing recovery" in freshness_contract["good_readiness_does_not_add_load_rule"]
    assert "10-15 W" in freshness_contract["recovery_execution_fallback_rule"]
    assert "3-5 evidence bullets" in freshness_contract["brevity_rule"]
