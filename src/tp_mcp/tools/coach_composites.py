"""Coach-oriented composite tools for Fitnessbot.

Read-only wrappers around existing TrainingPeaks tools plus pure guardrail helpers.
Do not add create/update/delete behavior here.
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
from datetime import date, timedelta
from typing import Any

from pydantic import ValidationError

from tp_mcp.coach_composite_helpers import (
    classify_missed_workout_reason,
    classify_readiness_snapshot,
    metric_value,
    normalize_subjective_feedback,
    summarize_feedback_patterns,
    validate_week_plan_guardrails,
    week_bounds,
)
from tp_mcp.tools._validation import SingleDateInput, format_validation_error
from tp_mcp.tools.events import tp_get_availability, tp_list_notes
from tp_mcp.tools.fitness import tp_get_fitness
from tp_mcp.tools.metrics import tp_get_metrics
from tp_mcp.tools.settings import tp_get_athlete_settings_summary
from tp_mcp.tools.weekly_summary import tp_get_weekly_summary
from tp_mcp.tools.workouts import tp_get_workout, tp_get_workout_note, tp_get_workouts


def _metric_date(metric: dict[str, Any]) -> str:
    """Best-effort ISO date extraction for TrainingPeaks health metric rows."""
    for key in (
        "date",
        "day",
        "metricDate",
        "metric_date",
        "workoutDay",
        "startDate",
        "start_time",
        "timestamp",
        "timeStamp",
        "created",
    ):
        value = metric.get(key)
        if value:
            return str(value)[:10]
    return ""


def _latest_metric_dict(metrics_result: dict[str, Any]) -> dict[str, Any]:
    metrics = metrics_result.get("metrics") or []
    if not metrics:
        return {}
    latest = metrics[-1]
    if isinstance(latest, dict):
        out = dict(latest)
        details = latest.get("details")
        if isinstance(details, list):
            for detail in details:
                label = detail.get("label") or detail.get("name") or detail.get("type")
                if label:
                    out[str(label).lower().replace(" ", "_")] = detail.get("value")
        return out
    return {}


def _metric_dict_for_date(metrics_result: dict[str, Any], ref_s: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the metric row for ``ref_s`` and explicit freshness metadata.

    TrainingPeaks overnight Garmin metrics can arrive after Martín asks for the
    morning brief. Stale fallback data may be shown as context, but it must not
    be treated as today's readiness.
    """
    metrics = [m for m in metrics_result.get("metrics", []) if isinstance(m, dict)]
    if not metrics:
        return {}, {
            "requested_date": ref_s,
            "source_date": None,
            "status": "missing",
            "is_current_day": False,
            "decision_rule": "do_not_use_stale_readiness_as_today",
        }

    same_day = [m for m in metrics if _metric_date(m) == ref_s]
    if same_day:
        return _latest_metric_dict({"metrics": same_day}), {
            "requested_date": ref_s,
            "source_date": ref_s,
            "status": "current",
            "is_current_day": True,
            "decision_rule": "ok_to_use_for_today_readiness",
        }

    dated = [(m, _metric_date(m)) for m in metrics]
    dated = [(m, d) for m, d in dated if d]
    dated_before_or_equal = [(m, d) for m, d in dated if d <= ref_s]
    if dated:
        fallback, fallback_date = dated_before_or_equal[-1] if dated_before_or_equal else dated[-1]
    else:
        fallback, fallback_date = metrics[-1], ""
    return _latest_metric_dict({"metrics": [fallback]}), {
        "requested_date": ref_s,
        "source_date": fallback_date or None,
        "status": "stale" if fallback_date else "unknown_date",
        "is_current_day": False,
        "decision_rule": "do_not_use_stale_readiness_as_today",
    }


def _body_battery_interpretation(latest_metrics: dict[str, Any]) -> dict[str, Any] | None:
    raw = latest_metrics.get("body_battery")
    if raw is None:
        raw = latest_metrics.get("Body Battery")
    if raw is None:
        return None

    interpretation: dict[str, Any] = {
        "raw": raw,
        "source_semantics": "TrainingPeaks Garmin Body Battery metric",
        "coaching_rule": (
            "Use as recovery-support evidence, not as permission to add training load. "
            "The max value of the array IS the wake/end-of-sleep value by Garmin's design "
            "(per Garmin official: Body Battery is fullest in the morning when you wake up; "
            "TP syncs 'the highest and lowest value for each day'). Use the max as the primary "
            "readiness number. Frame it as recovery support, not permission to add load."
        ),
    }
    if isinstance(raw, list):
        nums = [metric_value({"value": v}, "value") for v in raw]
        nums = [v for v in nums if v is not None]
        if len(nums) >= 3:
            bb_min = round(nums[0], 1)
            bb_max = round(nums[1], 1)
            bb_avg = round(nums[2], 1)
            interpretation.update({
                "format": "array_min_max_avg",
                "min": bb_min,
                "max": bb_max,
                "avg": bb_avg,
                "wake_value": bb_max,
                "display_guidance": (
                    f"Body Battery: {bb_max:.0f} al despertar "
                    f"(mínimo del día: {bb_min:.0f}, promedio: {bb_avg:.0f}); "
                    "señal de recuperación, no permiso para sumar carga."
                ),
            })
            # Readiness support level based on wake value (max)
            if bb_max >= 80:
                interpretation["recovery_support"] = "strong"
            elif bb_max >= 50:
                interpretation["recovery_support"] = "moderate"
            else:
                interpretation["recovery_support"] = "low"
        elif nums:
            interpretation.update({"format": "array_values", "values": [round(v, 1) for v in nums]})
    else:
        value = metric_value({"body_battery": raw}, "body_battery")
        if value is not None:
            interpretation.update({
                "format": "single_value",
                "value": round(value, 1),
                "wake_value": round(value, 1),
                "display_guidance": (
                    f"Body Battery: {value:.0f}; señal de recuperación, no permiso para sumar carga."
                ),
            })
            if value >= 80:
                interpretation["recovery_support"] = "strong"
            elif value >= 50:
                interpretation["recovery_support"] = "moderate"
            else:
                interpretation["recovery_support"] = "low"
    return interpretation


def _workouts_list(result: Any) -> list[dict[str, Any]]:
    if isinstance(result, dict):
        raw = result.get("workouts") or result.get("items") or []
    elif isinstance(result, list):
        raw = result
    else:
        raw = []
    return [item for item in raw if isinstance(item, dict)]


def _workout_id(workout: dict[str, Any]) -> str | None:
    for key in ("id", "workout_id", "workoutId", "workoutID"):
        value = workout.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return None


def _workout_date(workout: dict[str, Any]) -> str:
    return str(
        workout.get("date")
        or workout.get("workoutDay")
        or workout.get("startDate")
        or workout.get("start_time")
        or ""
    )[:10]


def _is_workout_completed(workout: dict[str, Any]) -> bool:
    completed = workout.get("completed")
    if completed is True:
        return True
    if completed is False:
        return False
    actual_fields = (
        "duration_actual",
        "actual_duration",
        "tss_actual",
        "actual_tss",
        "distance_actual",
        "actual_distance",
    )
    metrics = workout.get("metrics")
    metrics = metrics if isinstance(metrics, dict) else {}
    for key in actual_fields:
        if workout.get(key) not in (None, "", 0, 0.0):
            return True
        if metrics.get(key) not in (None, "", 0, 0.0):
            return True
    return False


def _is_planned_workout(workout: dict[str, Any]) -> bool:
    if _is_workout_completed(workout):
        return False
    if workout.get("planned") is True or workout.get("isPlanned") is True:
        return True
    if workout.get("type") and str(workout.get("type")).lower() in {"planned", "workout", "strength"}:
        return True
    planned_fields = ("duration_planned", "planned_duration", "tss_planned", "planned_tss", "distance_planned")
    metrics = workout.get("metrics")
    metrics = metrics if isinstance(metrics, dict) else {}
    return any(
        workout.get(k) not in (None, "", 0, 0.0) or metrics.get(k) not in (None, "", 0, 0.0)
        for k in planned_fields
    )


def _daily_brief_tss_contract(week_context: Any) -> dict[str, Any]:
    """Pre-compute TSS wording so Daily Briefs don't mislabel mixed totals.

    The daily brief often needs three distinct numbers: completed actual TSS,
    original planned TSS, and projected week TSS if the remaining planned
    sessions are completed. Returning them explicitly prevents the LLM from
    calling an actual+future projection "planificado total".
    """
    summary = week_context.get("weekly_summary", {}) if isinstance(week_context, dict) else {}
    workouts = summary.get("workouts", []) if isinstance(summary, dict) else []
    workouts = [item for item in workouts if isinstance(item, dict)]
    completed_actual = 0.0
    planned_original = 0.0
    projected_if_remaining_completed = 0.0
    for workout in workouts:
        planned_tss = _field_float(workout, "tss_planned", "planned_tss", "plannedTss")
        actual_tss = _field_float(workout, "tss_actual", "actual_tss", "tssActual")
        fallback_tss = _field_float(workout, "tss")
        if planned_tss is not None:
            planned_original += planned_tss
        elif fallback_tss is not None:
            planned_original += fallback_tss
        if _is_workout_completed(workout):
            value = actual_tss if actual_tss is not None else fallback_tss
            completed_actual += value or 0.0
            projected_if_remaining_completed += value or 0.0
        else:
            value = planned_tss if planned_tss is not None else fallback_tss
            projected_if_remaining_completed += value or 0.0
    return {
        "completed_actual_tss": round(completed_actual, 1),
        "planned_original_tss": round(planned_original, 1),
        "projected_tss_if_remaining_completed": round(projected_if_remaining_completed, 1),
        "wording_rule": (
            "Use 'completado' for completed_actual_tss, 'planificado original' for planned_original_tss, "
            "and 'proyectado si se completa lo restante' for projected_tss_if_remaining_completed. "
            "Never call a mixed actual+future projection 'planificado total'."
        ),
    }


async def tp_coach_week_context(week_of: str | None = None) -> dict[str, Any]:
    """Read-only composite weekly context for Fitnessbot reports."""
    try:
        ref = SingleDateInput(date=week_of).date if week_of else date.today()
    except (ValidationError, ValueError) as e:
        msg = format_validation_error(e) if isinstance(e, ValidationError) else str(e)
        return {"isError": True, "error_code": "VALIDATION_ERROR", "message": msg}

    start, end = week_bounds(ref)
    start_s, end_s = start.isoformat(), end.isoformat()

    tasks = {
        "weekly_summary": tp_get_weekly_summary(week_of=ref.isoformat()),
        "workouts": tp_get_workouts(start_s, end_s),
        "fitness": tp_get_fitness(days=7, start_date=start_s, end_date=end_s),
        "settings_summary": tp_get_athlete_settings_summary(),
        "metrics": tp_get_metrics(start_s, end_s),
        "notes": tp_list_notes(start_s, end_s),
        "availability": tp_get_availability(start_s, end_s),
    }
    results = await asyncio.gather(*tasks.values(), return_exceptions=True)
    data: dict[str, Any] = dict(zip(tasks.keys(), results, strict=True))
    missing: list[str] = []
    for key, value in list(data.items()):
        if isinstance(value, Exception):
            data[key] = {"isError": True, "message": str(value)}
            missing.append(key)
        elif isinstance(value, dict) and value.get("isError"):
            missing.append(key)

    data["week"] = {"start": start_s, "end": end_s}
    data["daily_brief_protocol"] = {
        "required_context": [
            "today_workout_and_objective",
            "yesterday_planned_vs_completed_with_workout_comment_private_note_and_calendar_notes",
            "week_to_date_planned_vs_completed_sessions_and_tss",
            "missed_extra_or_overcooked_workouts",
            "week_to_date_intensity_distribution_and_running_volume",
            "last_truly_completed_key_session",
            "remaining_week_key_sessions_and_availability",
            "readiness_metric_freshness_with_affected_signals",
        ],
        "decision_rule": (
            "Base daily recommendation on yesterday + week-to-date + remaining microcycle; "
            "do not compensate missed TSS automatically."
        ),
        "readiness_freshness_contract": {
            "if_stale_partial_or_missing": (
                "Say explicitly whether same-day readiness data is absent, stale, partial, or timestamp-not-confirmed; "
                "name affected signals such as HRV, sleep, pulse/RHR-proxy, Body Battery, and stress; "
                "do not cite fallback values as today's readiness; make a conservative load-based decision."
            ),
            "body_battery_rule": (
                "If Body Battery is a range or timeline, show range/average explicitly when available "
                "(e.g. rango TP 63→100, promedio ~90), distinguish sleep-start/pre-sleep, overnight "
                "recovery, and wake/end-of-sleep value, and frame it as support rather than permission to add load."
            ),
            "load_language_rule": (
                "Describe TSB/load with calibrated language: TSB around -10 after a long run is expected "
                "functional fatigue, not an alarm; recovery movement is OK, extra stimulus is not."
            ),
            "combined_readiness_load_rule": (
                "If HRV/sleep/pulse/Body Battery are good but recent load/TSB is elevated, explicitly say "
                "systemic recovery is good while legs/impact and microcycle load remain the limiter. Use a "
                "brief combined read rather than repeating metrics: good physiology + elevated load = "
                "execute easy, no extra."
            ),
            "two_light_rule": (
                "Separate physiological readiness from the session/microcycle decision in the headline when useful. "
                "For rest/recovery/no-workout days, prefer 'Readiness fisiológica: verde / Decisión: "
                "descanso planificado' "
                "instead of an ambiguous standalone '🟢 Hoy'."
            ),
            "good_readiness_does_not_add_load_rule": (
                "Good readiness authorizes executing recovery; it does not authorize extending, chasing pace, "
                "or adding training load."
            ),
            "recovery_execution_fallback_rule": (
                "For recovery sessions, include HR/breathing and RPE limiters: if RPE drifts to 3/10 sustained, "
                "or HR/breathing feels high for the target power, reduce 10-15 W or walk."
            ),
            "brevity_rule": (
                "Routine Daily Briefs should stay compact: decision, 3-5 evidence bullets, execution caps, "
                "and report prompt; expand only for safety/data-quality/plan-change reasons."
            ),
        },
    }
    data["missing_or_error_sources"] = missing
    data["safety"] = {
        "read_only": True,
        "writes_allowed": False,
        "message": "Composite context only; create/update/delete still require explicit approval and readback.",
    }
    return data


def _field_float(workout: dict[str, Any], *keys: str) -> float | None:
    metrics = workout.get("metrics")
    metrics = metrics if isinstance(metrics, dict) else {}
    for key in keys:
        value = workout.get(key)
        if value is None:
            value = metrics.get(key)
        if value in (None, ""):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _workout_title(workout: dict[str, Any]) -> str:
    return str(workout.get("title") or workout.get("workoutTitle") or workout.get("name") or "")


def _is_key_period_workout(workout: dict[str, Any]) -> bool:
    title = _workout_title(workout).lower()
    sport = str(workout.get("sport") or workout.get("workout_type") or workout.get("type") or "").lower()
    duration = (
        _field_float(workout, "duration_actual", "actual_duration", "duration_planned", "planned_duration") or 0.0
    )
    key_terms = (
        "long", "largo", "larga", "tirada", "tempo", "threshold", "umbral",
        "vo2", "interval", "intervalo", "series", "fartlek", "test", "race", "carrera",
    )
    return any(term in title for term in key_terms) or (
        duration >= 75 and ("run" in sport or "bike" in sport or sport in {"completed", "workout"})
    )


def _has_plan_actual_anomaly(workout: dict[str, Any]) -> bool:
    planned_tss = _field_float(workout, "tss_planned", "planned_tss")
    actual_tss = _field_float(workout, "tss_actual", "actual_tss")
    planned_duration = _field_float(workout, "duration_planned", "planned_duration")
    actual_duration = _field_float(workout, "duration_actual", "actual_duration")
    for planned, actual in ((planned_tss, actual_tss), (planned_duration, actual_duration)):
        if planned and actual is not None and abs(actual - planned) / max(planned, 1.0) >= 0.25:
            return True
    return False


def _period_expansion_reasons(workout: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    workout_day = _workout_date(workout)
    is_due = not workout_day or workout_day <= date.today().isoformat()
    if is_due and _is_planned_workout(workout):
        reasons.append("planned_not_completed_or_missing_actuals")
    if _is_key_period_workout(workout):
        reasons.append("key_session")
    if _has_plan_actual_anomaly(workout):
        reasons.append("plan_actual_anomaly")
    return reasons


async def tp_coach_daily_brief_context(date_str: str | None = None) -> dict[str, Any]:
    """Read-only daily brief preflight with mandatory missed-yesterday detail expansion.

    If yesterday has a planned-but-not-completed workout, this composite opens the
    workout detail plus private note before returning, so the coach cannot invent
    a reason from the list view.
    """
    try:
        ref = SingleDateInput(date=date_str).date if date_str else date.today()
    except (ValidationError, ValueError) as e:
        msg = format_validation_error(e) if isinstance(e, ValidationError) else str(e)
        return {"isError": True, "error_code": "VALIDATION_ERROR", "message": msg}

    yesterday = ref - timedelta(days=1)
    start, end = week_bounds(ref)
    yesterday_s = yesterday.isoformat()
    ref_s = ref.isoformat()

    tasks = {
        "readiness": tp_coach_readiness_snapshot(date_str=ref_s),
        "week_context": tp_coach_week_context(week_of=ref_s),
        "workouts_window": tp_get_workouts(yesterday_s, end.isoformat()),
        "calendar_notes": tp_list_notes(yesterday_s, ref_s),
    }
    results = await asyncio.gather(*tasks.values(), return_exceptions=True)
    data = dict(zip(tasks.keys(), results, strict=True))
    missing: list[str] = []
    for key, value in list(data.items()):
        if isinstance(value, Exception):
            data[key] = {"isError": True, "message": str(value)}
            missing.append(key)
        elif isinstance(value, dict) and value.get("isError"):
            missing.append(key)

    window_workouts = _workouts_list(data.get("workouts_window"))
    today_workouts = [w for w in window_workouts if _workout_date(w) == ref_s]
    yesterday_workouts = [w for w in window_workouts if _workout_date(w) == yesterday_s]
    calendar_notes_result = data.get("calendar_notes")
    calendar_notes = calendar_notes_result.get("notes", []) if isinstance(calendar_notes_result, dict) else []

    expanded_yesterday: list[dict[str, Any]] = []
    detail_expansion_required = False
    detail_expansion_ok = True
    verification: list[str] = []

    for workout in yesterday_workouts:
        wid = _workout_id(workout)
        missed = _is_planned_workout(workout)
        item: dict[str, Any] = {
            "id": wid,
            "title": workout.get("title") or workout.get("workoutTitle") or workout.get("name"),
            "date": _workout_date(workout),
            "list_view": workout,
            "status": "missed_or_uncompleted" if missed else "completed_or_not_planned",
        }
        if missed:
            detail_expansion_required = True
            detail: Any = {"isError": True, "message": "missing_workout_id"}
            private_note: Any = {"isError": True, "message": "missing_workout_id"}
            if wid:
                detail, private_note = await asyncio.gather(
                    tp_get_workout(workout_id=wid),
                    tp_get_workout_note(workout_id=wid),
                    return_exceptions=True,
                )
            if isinstance(detail, Exception):
                detail = {"isError": True, "message": str(detail)}
            if isinstance(private_note, Exception):
                private_note = {"isError": True, "message": str(private_note)}
            if isinstance(detail, dict) and not detail.get("isError"):
                verification.append("tp_get_workout_detail_and_private_note_called_for_missed_yesterday")
                reason = classify_missed_workout_reason(
                    detail,
                    private_note=private_note if isinstance(private_note, dict) else {},
                    calendar_notes=calendar_notes if isinstance(calendar_notes, list) else [],
                )
            else:
                detail_expansion_ok = False
                reason = {
                    "category": "unknown",
                    "confidence": "blocked",
                    "evidence": "",
                    "decision_rule": "do_not_compensate_missed_tss",
                }
            item.update({"detail": detail, "private_note": private_note, "reason": reason})
        expanded_yesterday.append(item)

    can_interpret = (not detail_expansion_required) or detail_expansion_ok
    readiness_result = data.get("readiness")
    readiness_freshness = readiness_result.get("metric_freshness", {}) if isinstance(readiness_result, dict) else {}
    readiness_is_current = readiness_freshness.get("is_current_day") is True
    week_context = data.get("week_context")
    return {
        "date": ref_s,
        "week": {"start": start.isoformat(), "end": end.isoformat()},
        "daily_brief_output_contract": {
            "headline_rule": (
                "Separate Readiness fisiológica from Decisión del día/sesión; never use ambiguous "
                "'🟢 Hoy' alone."
            ),
            "rest_day_rule": (
                "If today has no planned workout, say descanso planificado/sin sesión and do not imply "
                "green means add training."
            ),
            "closure_required_when_space_allows": [
                "Confianza",
                "Qué voy a vigilar",
                "Dato faltante",
                "Qué cambiaría la decisión",
            ],
            "recovery_wording_rule": (
                "Avoid absolutes like 'plenamente recuperado'; say physiological signals show strong recovery "
                "and still respect the microcycle."
            ),
            "body_battery_format_rule": (
                "If Body Battery is [min,max,avg], prefer: max al despertar, mínimo, promedio; "
                "recovery support, not permission to add load."
            ),
        },
        "tss_language_contract": _daily_brief_tss_contract(week_context),
        "today": today_workouts,
        "yesterday": expanded_yesterday,
        "readiness": data.get("readiness"),
        "week_context": week_context,
        "calendar_notes": calendar_notes_result,
        "decision_guardrails": {
            "can_interpret_missed_yesterday": can_interpret,
            "can_use_readiness_metrics_for_today": readiness_is_current,
            "readiness_metric_freshness": readiness_freshness,
            "must_not_guess_missed_reason_from_list_view": True,
            "must_not_use_stale_readiness_as_today": not readiness_is_current,
            "must_not_compensate_missed_tss_automatically": True,
            "if_false": "Say confidence is limited and fetch workout detail/private note before recommendation.",
            "if_readiness_stale_or_missing": (
                "Do not cite fallback HRV/sleep/pulse/Body Battery as today's data. Label same-day readiness as "
                "pending/stale and make a conservative load-based recommendation until current metrics arrive."
            ),
        },
        "verification": verification,
        "missing_or_error_sources": missing,
        "safety": {"read_only": True, "writes_allowed": False},
    }


# ── Compact summary builders ────────────────────────────────────────────────
# These functions produce small, pre-computed summaries that the LLM can use
# directly for narrative without parsing 300K+ chars of raw tool output.
# Inspired by the "summary + drill-down" pattern from MCP community best
# practices (GitHub discussions, Context Mode, arxiv:2511.22729): the tool
# returns a compact summary up front and keeps full data available for
# drill-down. This prevents context overflow from hiding recent workouts.


# Map workoutTypeValueId to sport name (verified from TP API samples).
_SPORT_MAP = {2: "Bike", 3: "Run", 1: "Swim", 9: "Strength", 100: "Other", 10: "Note", 7: "Rest", 13: "Other2"}


def _sport_name(type_id: Any) -> str:
    return _SPORT_MAP.get(type_id, f"Type{type_id}")


def _detect_sport_from_expanded(ew: dict[str, Any]) -> str:
    """Detect sport from an expanded workout dict.

    Expanded workouts store the raw workout under 'list_view', which has
    the sport/workoutTypeValueId fields. Also check top-level for safety.
    """
    # Check top-level first
    sport = ew.get("sport")
    if isinstance(sport, str) and sport:
        return sport
    type_id = ew.get("workoutTypeValueId") or ew.get("workout_type_value_id")
    if type_id is not None:
        return _sport_name(type_id)
    # Check inside list_view (the raw workout object)
    list_view = ew.get("list_view")
    if isinstance(list_view, dict):
        lv_sport = list_view.get("sport")
        if isinstance(lv_sport, str) and lv_sport:
            return lv_sport
        lv_type_id = list_view.get("workoutTypeValueId") or list_view.get("workout_type_value_id")
        if lv_type_id is not None:
            return _sport_name(lv_type_id)
    return "Unknown"


def _build_workout_summary(
    raw_workouts_result: Any,
    expanded_workouts: list[dict[str, Any]],
    period_days: int | None = None,
) -> dict[str, Any]:
    """Build a compact summary of workouts for the LLM narrative.

    This replaces the need for the LLM to parse 54+ raw workouts and 26+
    expanded workouts (147K+ chars). Instead it gets:
    - Total counts and TSS by sport
    - Long run progression with dates, durations, distances, TSS
    - Weekly TSS with dates, sport breakdown, and session counts
    - Key milestones (new maximums, firsts, notable sessions)
    - RPE/feeling highlights from expanded workouts
    All in ~2-3K chars instead of 147K.
    """
    workouts = _workouts_list(raw_workouts_result)

    # Sport detection: the normalized workout format from tp_get_workouts has a
    # "sport" string field ("Run", "Bike", "Strength", etc.). The raw TP API
    # format has workoutTypeValueId (int). Support both.
    def _detect_sport(w: dict[str, Any]) -> str:
        sport = w.get("sport")
        if isinstance(sport, str) and sport:
            return sport
        type_id = w.get("workoutTypeValueId")
        return _sport_name(type_id)

    # Duration detection: normalized format uses duration_actual (hours decimal),
    # raw API uses totalTime (hours decimal). Both are in hours.
    def _get_duration_min(w: dict[str, Any]) -> float:
        dur_h = _field_float(w, "duration_actual", "durationActual", "totalTime") or 0
        return dur_h * 60 if dur_h < 10 else dur_h  # TP stores hours as decimal

    # Filter out spurious sessions: Garmin sync artifacts (Other sport, <5 TSS, <3 min)
    # These contaminate counts and narratives (rule #49). Auto-filter in code.
    def _is_spurious(w: dict[str, Any]) -> bool:
        sport = _detect_sport(w)
        if sport != "Other":
            return False
        tss = _field_float(w, "tssActual", "tss_actual", "tss") or 0
        dur_min = _get_duration_min(w)
        return tss < 5 and dur_min < 3

    completed = [
        w for w in workouts
        if _field_float(w, "tssActual", "tss_actual", "tss")
        and (_field_float(w, "tssActual", "tss_actual", "tss") or 0) > 0
        and not _is_spurious(w)
    ]

    # Distance detection: normalized format uses distance_actual_km (already km),
    # raw API uses distance (meters). Detect by magnitude.
    def _get_distance_km(w: dict[str, Any]) -> float:
        dist = _field_float(w, "distance_actual_km", "distanceActual", "distance", "distance_actual") or 0
        if dist > 100:  # meters — convert
            return dist / 1000
        return dist

    # Sport breakdown
    by_sport: dict[str, dict[str, float]] = {}
    for w in completed:
        sport = _detect_sport(w)
        tss = _field_float(w, "tssActual", "tss_actual", "tss") or 0
        dur_min = _get_duration_min(w)
        bucket = by_sport.setdefault(sport, {"sessions": 0.0, "tss": 0.0, "duration_h": 0.0})
        bucket["sessions"] += 1
        bucket["tss"] += float(tss)
        bucket["duration_h"] += dur_min / 60

    sport_summary = {}
    for sport, v in sorted(by_sport.items(), key=lambda x: -x[1]["tss"]):
        sport_summary[sport] = {
            "sessions": int(v["sessions"]),
            "tss": round(v["tss"], 1),
            "duration_hours": round(v["duration_h"], 1),
        }

    # Long run progression (Run sport, >=45 min, sorted by date)
    # Enriched with RPE + Feeling from expanded_workouts (cross-reference by date+sport)
    # Key: use (date, sport) tuple to avoid cross-contamination when multiple
    # workouts exist on the same date (e.g., Run + Strength on same day).
    expanded_by_date_sport: dict[tuple[str, str], dict] = {}
    for ew in expanded_workouts:
        d = ew.get("date")
        s = _detect_sport_from_expanded(ew)
        if d:
            expanded_by_date_sport[(d, s)] = ew

    long_runs = []
    for w in completed:
        sport = _detect_sport(w)
        if sport != "Run":
            continue
        dur_min = _get_duration_min(w)
        if dur_min >= 45:
            d = _workout_date(w)
            # Cross-reference with expanded_workouts for RPE/Feeling
            # Use (date, sport) key to avoid matching wrong sport on same-day workouts
            ew = expanded_by_date_sport.get((d, "Run"), {})
            sf = ew.get("subjective_feedback") or {}
            rpe = sf.get("rpe") if isinstance(sf, dict) else None
            feeling = sf.get("feeling") if isinstance(sf, dict) else None
            entry: dict[str, Any] = {
                "date": d,
                "duration_min": round(dur_min),
                "distance_km": round(_get_distance_km(w), 1),
                "tss": round(_field_float(w, "tssActual", "tss_actual", "tss") or 0, 1),
                "title": _workout_title(w),
            }
            # Add RPE/Feeling if available (may be None for older sessions)
            if rpe is not None:
                entry["rpe"] = rpe
            if feeling and isinstance(feeling, dict):
                entry["feeling"] = feeling.get("label", "")
                entry["feeling_score_1_to_5"] = feeling.get("score_1_to_5")
            elif feeling and isinstance(feeling, str):
                entry["feeling"] = feeling
            long_runs.append(entry)
    long_runs.sort(key=lambda x: x["date"])

    # Find new maximums
    max_dur = 0
    milestones = []
    for lr in long_runs:
        if lr["duration_min"] > max_dur:
            max_dur = lr["duration_min"]
            if max_dur > 0:
                milestones.append(
                    f"New long run max: {max_dur} min on {lr['date']} "
                    f"({lr['distance_km']} km, {lr['tss']} TSS)"
                )

    # Weekly TSS with dates and sport breakdown
    from datetime import date as date_cls
    from datetime import timedelta as td
    def week_start(d_str: str) -> str:
        d = date_cls.fromisoformat(d_str)
        ws = d - td(days=d.weekday())
        return ws.isoformat()

    weekly: dict[str, Any] = {}
    for w in completed:
        d = _workout_date(w)
        if not d:
            continue
        ws_key = week_start(d)
        tss = _field_float(w, "tssActual", "tss_actual", "tss") or 0
        sport = _detect_sport(w)
        bucket = weekly.setdefault(ws_key, {"tss": 0.0, "sessions": 0, "sports": {}})
        bucket["tss"] += float(tss)
        bucket["sessions"] += 1
        sports = bucket["sports"]
        sport_bucket = sports.setdefault(sport, {"count": 0, "tss": 0.0})
        sport_bucket["count"] += 1
        sport_bucket["tss"] += float(tss)

    weekly_table = []
    for ws_key in sorted(weekly.keys()):
        v = weekly[ws_key]
        we = (date_cls.fromisoformat(ws_key) + td(days=6)).isoformat()
        sport_str = ", ".join(
            f"{s}:{v2['count']}({v2['tss']:.0f})"
            for s, v2 in sorted(v["sports"].items(), key=lambda x: -x[1]["tss"])
        )
        weekly_table.append({
            "week_start": ws_key,
            "week_end": we,
            "week_label": f"{ws_key} to {we}",
            "tss": round(v["tss"], 1),
            "sessions": int(v["sessions"]),
            "sports": sport_str,
        })

    # RPE/feeling highlights from expanded workouts (compact)
    # Include all RPE >= 7 (high effort — behavioral flag for easy sessions)
    # and RPE <= 2 (very easy — good compliance signal)
    rpe_highlights = []
    for ew in expanded_workouts:
        sf = ew.get("subjective_feedback")
        if isinstance(sf, dict) and sf:
            rpe = sf.get("rpe")
            feeling = sf.get("feeling")
            if rpe and (rpe >= 7 or rpe <= 2):
                rpe_highlights.append({
                    "date": ew.get("date"),
                    "title": ew.get("title"),
                    "sport": _detect_sport_from_expanded(ew),
                    "rpe": rpe,
                    "feeling": feeling,
                })

    # Pre-compute RPE≥7 summary so the LLM doesn't have to count.
    # This prevents the recurring bug where the LLM reports "2 sessions"
    # when there are actually 3 (typically omitting Bike entries).
    rpe7_entries = [r for r in rpe_highlights if r.get("rpe", 0) >= 7]
    rpe7_by_sport: dict[str, int] = {}
    for r in rpe7_entries:
        sport_key = str(r.get("sport", "Unknown"))
        rpe7_by_sport[sport_key] = rpe7_by_sport.get(sport_key, 0) + 1
    rpe7_summary = {
        "total_count": len(rpe7_entries),
        "by_sport": dict(rpe7_by_sport),
        "dates": [r["date"] for r in rpe7_entries],
        "coaching_note": (
            f"There are {len(rpe7_entries)} sessions with RPE≥7 across "
            f"{len(rpe7_by_sport)} sport(s): {', '.join(f'{k}: {v}' for k, v in sorted(rpe7_by_sport.items()))}. "
            f"Report ALL of them — do not omit any sport. Dates: {', '.join(r['date'] for r in rpe7_entries)}."
        ) if rpe7_entries else "No sessions with RPE≥7 in this period.",
    }

    # Key sessions (from expanded_workouts)
    key_sessions = []
    for ew in expanded_workouts:
        reasons = ew.get("expansion_reasons", [])
        if "key_session" in reasons:
            key_sessions.append({
                "date": ew.get("date"),
                "title": ew.get("title"),
                "id": ew.get("id"),
            })

    return {
        "total_sessions": len(completed),
        "session_frequency": {
            "sessions": len(completed),
            "period_days": period_days,
            "period_weeks": round(period_days / 7, 2) if period_days else None,
            "sessions_per_week": round(len(completed) / (period_days / 7), 1) if period_days else None,
            "display": (
                f"{len(completed)} sesiones en {period_days} días = "
                f"{len(completed) / (period_days / 7):.1f}/sem"
            ) if period_days else None,
            "instruction": "Use sessions_per_week/display directly — do NOT recalculate session frequency manually.",
        },
        "total_tss": round(sum(_field_float(w, "tssActual", "tss_actual", "tss") or 0 for w in completed), 1),
        "by_sport": sport_summary,
        "long_run_progression": long_runs,
        "long_run_max_duration_min": max_dur if long_runs else 0,
        "long_run_max_date": long_runs[-1]["date"] if long_runs else "",
        "milestones": milestones,
        "weekly_tss": weekly_table,
        "weekly_breakdown": weekly_table,  # alias for SKILL.md rule #42
        "key_sessions": key_sessions,
        "rpe_extremes": rpe_highlights,
        "rpe7_summary": rpe7_summary,
        "instruction": (
            "Use this workout_summary for all workout narrative. "
            "Long run progression, weekly TSS with dates, sport breakdown, milestones, "
            "and session_frequency are pre-computed — do NOT re-derive from raw workouts or expanded_workouts. "
            "Use session_frequency.display for sessions/week; do NOT calculate it manually. "
            "If you need RPE/feeling/comments for a specific session, look up by id in expanded_workouts. "
            "RPE≥7 count is pre-computed in rpe7_summary — use that count, do NOT count manually."
        ),
    }


def _compact_metrics(metrics_result: Any) -> dict[str, Any]:
    """Summarize Garmin health metrics into a compact structure.

    The raw metrics response can be 128K+ chars (73 days × multiple metrics
    × detail arrays). This function computes averages, ranges, and latest
    values for HRV, sleep, pulse/RHR, Body Battery, and stress — everything
    the LLM needs for readiness narrative in ~1K chars.
    """
    if not isinstance(metrics_result, dict):
        return {"isError": True, "message": "metrics result is not a dict"}

    metrics = metrics_result.get("metrics", [])
    if not metrics:
        return {"available": False, "reason": "no metrics returned"}

    # Extract latest values and compute averages for key signals
    from collections import defaultdict
    signal_values = defaultdict(list)
    latest_metric = {}
    latest_date = ""

    for m in metrics:
        if not isinstance(m, dict):
            continue
        d = _metric_date(m)
        if d and d > latest_date:
            latest_date = d
            latest_metric = _latest_metric_dict({"metrics": [m]})

        details = m.get("details", [])
        if isinstance(details, list):
            for detail in details:
                label = (detail.get("label") or detail.get("name") or "").lower().replace(" ", "_")
                value = detail.get("value")
                if value is not None:
                    signal_values[label].append(value)

    def avg(key: str) -> float | None:
        vals = signal_values.get(key, [])
        if not vals:
            # Try key with suffix (e.g. "sleep" → "sleep_hours", "stress" → "stress_level")
            for full_key in signal_values:
                if full_key.startswith(key):
                    vals = signal_values[full_key]
                    break
        if not vals:
            return None
        # Handle array values (e.g. Stress Level = [min, max, avg])
        # Use the avg (last element) for averaging
        scalar_vals = []
        for v in vals:
            if isinstance(v, list) and len(v) >= 3:
                scalar_vals.append(v[-1])  # avg is last element
            elif isinstance(v, (int, float)):
                scalar_vals.append(v)
        if not scalar_vals:
            return None
        return round(sum(scalar_vals) / len(scalar_vals), 1)

    def latest(key: str) -> float | None:
        v = latest_metric.get(key)
        if v is None:
            # Try key with suffix (e.g. "sleep" → "sleep_hours", "stress" → "stress_level")
            for full_key in latest_metric:
                if full_key.startswith(key) and full_key != key:
                    v = latest_metric[full_key]
                    break
        if v is not None:
            # Handle array values (e.g. Stress Level = [min, max, avg])
            if isinstance(v, list) and len(v) >= 3:
                return round(float(v[-1]), 1)  # avg is last element
            try:
                return round(float(v), 1)
            except (TypeError, ValueError):
                pass
        return None

    # Body Battery from latest metric
    bb = _body_battery_interpretation(latest_metric)

    # Stress Level interpretation (array format: [min, max, avg])
    stress_raw = latest_metric.get("stress_level")
    stress_detail = None
    if isinstance(stress_raw, list) and len(stress_raw) >= 3:
        stress_detail = {
            "raw": stress_raw,
            "format": "array_min_max_avg",
            "min": stress_raw[0],
            "max": stress_raw[1],
            "avg": stress_raw[2],
            "display": f"Stress: avg {stress_raw[2]} (max {stress_raw[1]})",
        }

    # Pre-formatted readiness snapshot — a concise string with all 5 signals.
    # The LLM can use it verbatim, partially, or as a quick reference.
    # This is a FORMATTED FACT, not a prescription — the LLM decides which
    # signals are relevant per question type.
    hrv_l = latest("hrv")
    hrv_a = avg("hrv")
    sleep_l = latest("sleep")
    sleep_a = avg("sleep")
    rhr_l = latest("pulse")
    rhr_a = avg("pulse")
    bb_display = bb.get("display_guidance", "") if bb else ""
    stress_display = stress_detail.get("display", "") if stress_detail else ""

    readiness_parts = []
    if hrv_l is not None:
        readiness_parts.append(f"HRV {hrv_l} (avg {hrv_a})")
    if sleep_l is not None:
        readiness_parts.append(f"Sueño {sleep_l}h (avg {sleep_a}h)")
    if rhr_l is not None:
        readiness_parts.append(f"RHR {rhr_l} bpm (avg {rhr_a})")
    if bb_display:
        readiness_parts.append(bb_display)
    if stress_display:
        readiness_parts.append(stress_display)
    readiness_snapshot = " | ".join(readiness_parts) if readiness_parts else "No readiness metrics available"

    return {
        "available": True,
        "latest_date": latest_date,
        "latest": {
            "hrv": hrv_l,
            "sleep_hours": sleep_l,
            "pulse_rhr": rhr_l,
            "body_battery": bb.get("wake_value") if bb else None,
            "stress": latest("stress"),
        },
        "averages": {
            "hrv": hrv_a,
            "sleep_hours": sleep_a,
            "pulse_rhr": rhr_a,
            "stress": avg("stress"),
        },
        "data_points": len(metrics),
        "body_battery_detail": bb,
        "stress_detail": stress_detail,
        "readiness_snapshot": readiness_snapshot,
        "instruction": (
            "Use metrics_compact for readiness narrative. "
            "Latest values are from the most recent metric date. "
            "Averages cover the full period. For day-by-day detail, use the 'metrics' field. "
            "Available readiness signals: (1) HRV (latest + average), "
            "(2) sleep_hours (latest + average), "
            "(3) pulse_rhr / RHR (latest + average), "
            "(4) body_battery — use body_battery_detail.display_guidance for the full format (wake value + min + avg), "
            "(5) stress — use stress_detail.display if available (includes avg + max). "
            "Choose which signals are relevant per the question type — a post-workout analysis "
            "of a strength session may not need stress level, while a global analysis should "
            "include all 5. Use your judgment, not a rigid checklist."
        ),
    }


def _compact_fitness(fitness_result: Any) -> dict[str, Any]:
    """Summarize the narrow-period fitness data (CTL/ATL/TSB daily arrays).

    The raw fitness response includes daily_data with 73+ entries (~5.5K).
    This function keeps only current values, 7-day rolling averages, and
    a compact weekly summary — enough for the LLM to narrate trends
    without 5K of daily arrays.
    """
    if not isinstance(fitness_result, dict):
        return {"isError": True, "message": "fitness result is not a dict"}

    current = fitness_result.get("current", {})
    daily = fitness_result.get("daily_data", [])

    if not daily:
        return {
            "current": current,
            "available": False,
        }

    # Weekly CTL/ATL/TSB summary from daily_data
    from datetime import date as date_cls
    from datetime import timedelta as td

    weekly: dict[str, dict[str, list[float]]] = {}
    for entry in daily:
        d = entry.get("date", "")
        if not d:
            continue
        try:
            dt = date_cls.fromisoformat(d)
        except (ValueError, TypeError):
            continue
        ws = (dt - td(days=dt.weekday())).isoformat()
        bucket = weekly.setdefault(ws, {"ctl": [], "atl": [], "tsb": [], "tss": []})
        bucket["ctl"].append(float(entry.get("ctl") or 0))
        bucket["atl"].append(float(entry.get("atl") or 0))
        bucket["tsb"].append(float(entry.get("tsb") or 0))
        bucket["tss"].append(float(entry.get("tss") or 0))

    weekly_summary = []
    for ws in sorted(weekly.keys()):
        v = weekly[ws]
        weekly_summary.append({
            "week_start": ws,
            "ctl_avg": round(sum(v["ctl"]) / max(len(v["ctl"]), 1), 1),
            "atl_avg": round(sum(v["atl"]) / max(len(v["atl"]), 1), 1),
            "tsb_avg": round(sum(v["tsb"]) / max(len(v["tsb"]), 1), 1),
            "tss_total": round(sum(v["tss"]), 1),
        })

    return {
        "current": current,
        "weekly_summary": weekly_summary,
        "data_points": len(daily),
        "instruction": (
            "Use fitness_compact for current CTL/ATL/TSB and weekly trends. "
            "For daily detail, call tp_get_fitness directly."
        ),
    }


async def _compute_fitness_historical() -> dict[str, Any]:
    """Fetch full fitness history from 2024-01-01 to today and compute key milestones.

    This ensures the LLM always has the real historical peak CTL, nadir, and
    monthly averages available — regardless of what date range it passed to
    tp_coach_period_review_context. The LLM cannot under-query or hallucinate
    a peak value because the code injects the ground truth.
    """
    hist_start = "2024-01-01"
    hist_end = date.today().isoformat()
    hist_days = (date.today() - date.fromisoformat(hist_start)).days

    # FitnessInput caps days at 365, but when using start_date/end_date
    # the days field is ignored — the API accepts arbitrary ranges via the
    # performedata endpoint.
    try:
        result = await tp_get_fitness(
            start_date=hist_start,
            end_date=hist_end,
        )
    except Exception as exc:
        return {"isError": True, "message": f"fitness_historical query failed: {exc}"}

    if isinstance(result, dict) and result.get("isError"):
        return result

    daily = result.get("daily_data", [])
    if not daily:
        return {"isError": True, "message": "No daily fitness data returned for historical range."}

    # Compute peak CTL and its date
    peak_ctl = 0.0
    peak_date = ""
    for entry in daily:
        ctl = entry.get("ctl", 0)
        if ctl > peak_ctl:
            peak_ctl = ctl
            peak_date = entry.get("date", "")

    # Compute nadir (minimum CTL after the peak)
    nadir_ctl = peak_ctl
    nadir_date = ""
    found_peak = False
    for entry in daily:
        d = entry.get("date", "")
        ctl = entry.get("ctl", 0)
        if d == peak_date:
            found_peak = True
        if found_peak and ctl < nadir_ctl:
            nadir_ctl = ctl
            nadir_date = d

    # Current values (last entry)
    current = daily[-1] if daily else {}
    current_ctl = current.get("ctl", 0)
    pct_of_peak = round(current_ctl / peak_ctl * 100, 1) if peak_ctl > 0 else 0

    # Monthly averages
    monthly: dict[str, dict[str, float]] = {}
    for entry in daily:
        d = entry.get("date", "")
        if not d:
            continue
        month = d[:7]
        if month not in monthly:
            monthly[month] = {"ctl_sum": 0.0, "count": 0, "ctl_max": 0.0}
        ctl = entry.get("ctl", 0)
        monthly[month]["ctl_sum"] += ctl
        monthly[month]["count"] += 1
        if ctl > monthly[month]["ctl_max"]:
            monthly[month]["ctl_max"] = ctl

    monthly_summary = {}
    for month in sorted(monthly.keys()):
        v = monthly[month]
        monthly_summary[month] = {
            "avg_ctl": round(v["ctl_sum"] / v["count"], 1) if v["count"] else 0,
            "max_ctl": round(v["ctl_max"], 1),
            "days": v["count"],
        }

    # Top 5 CTL peaks with dates
    sorted_by_ctl = sorted(daily, key=lambda x: x.get("ctl", 0), reverse=True)
    top5 = [
        {
            "date": e.get("date", ""),
            "ctl": e.get("ctl", 0),
            "atl": e.get("atl", 0),
            "tsb": e.get("tsb", 0),
        }
        for e in sorted_by_ctl[:5]
    ]

    return {
        "period": {"start": hist_start, "end": hist_end, "days": hist_days},
        "peak_ctl": round(peak_ctl, 1),
        "peak_date": peak_date,
        "nadir_after_peak": round(nadir_ctl, 1),
        "nadir_date": nadir_date,
        "current_ctl": round(current_ctl, 1),
        "pct_of_peak": pct_of_peak,
        "top_5_ctl_peaks": top5,
        "monthly_avg_ctl": monthly_summary,
        "total_data_points": len(daily),
        "coaching_instruction": (
            f"REAL HISTORICAL PEAK: CTL {round(peak_ctl, 1)} on {peak_date}. "
            f"Current CTL {round(current_ctl, 1)} = {pct_of_peak}% of peak. "
            f"NADIR after peak: CTL {round(nadir_ctl, 1)} on {nadir_date} — this is the lowest point after the peak, "
            f"quantifying the full extent of the detraining/stop period. "
            f"The trajectory from {round(peak_ctl, 1)} → {round(nadir_ctl, 1)} "
            f"→ {round(current_ctl, 1)} tells the complete story. "
            f"Use these values — do NOT guess, hallucinate, or cite any other peak value. "
            f"The {round(peak_ctl, 1)} comes from TrainingPeaks performedata API (2024-01-01 to today). "
            f"If any other source says a different peak, THIS value is the ground truth."
        ),
    }


def _comment_text_from_expanded(expanded_workout: dict[str, Any]) -> str:
    detail = expanded_workout.get("detail")
    parts: list[str] = []
    if isinstance(detail, dict):
        for key in ("workout_comments", "athlete_comment", "comments", "new_comment", "description"):
            value = detail.get(key)
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        parts.append(str(item.get("comment") or item.get("text") or ""))
                    else:
                        parts.append(str(item))
            elif value:
                parts.append(str(value))
    for key in ("title", "date"):
        value = expanded_workout.get(key)
        if value:
            parts.append(str(value))
    return " ".join(parts).strip()


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    lower = text.lower()
    return any(needle in lower for needle in needles)


def _build_science_guardrails(
    *,
    workout_summary: dict[str, Any],
    metrics_compact: dict[str, Any],
    fitness_compact: dict[str, Any],
    fitness_historical: dict[str, Any] | None,
    expanded_workouts: list[dict[str, Any]],
    period_days: int,
    cycle_status: dict[str, Any],
    pain_mentions: int,
    pw_hr_status: dict[str, Any],
    consistency_warnings: list[str],
) -> dict[str, Any]:
    """Evidence-based guardrails for elite-coach narratives.

    This is deliberately descriptive: it flags domains and uncertainty, but does
    not diagnose, prescribe, or decide progression.
    """
    comments = [_comment_text_from_expanded(ew) for ew in expanded_workouts]
    heat_terms = ("calor", "heat", "humedad", "humidity", "bochorno")
    fatigue_terms = ("fatiga", "fatigue", "cansancio", "agot", "sin energía", "sin energia")
    fueling_terms = ("gel", "carbo", "cho", "fuel", "hidrat", "sodio", "electrol")
    doms_terms = ("doms", "agujeta", "adolorido", "dolor muscular")
    heat_mentions = sum(1 for text in comments if _contains_any(text, heat_terms))
    fatigue_mentions = sum(1 for text in comments if _contains_any(text, fatigue_terms))
    fueling_mentions = sum(1 for text in comments if _contains_any(text, fueling_terms))
    doms_mentions = sum(1 for text in comments if _contains_any(text, doms_terms))

    rpe7_summary = workout_summary.get("rpe7_summary", {}) if isinstance(workout_summary, dict) else {}
    rpe7_count = int(rpe7_summary.get("total_count") or 0) if isinstance(rpe7_summary, dict) else 0
    session_frequency = workout_summary.get("session_frequency", {}) if isinstance(workout_summary, dict) else {}
    sessions_per_week = session_frequency.get("sessions_per_week") if isinstance(session_frequency, dict) else None
    weekly_tss = workout_summary.get("weekly_tss", []) if isinstance(workout_summary, dict) else []
    weekly_tss_values = [float(w.get("tss") or 0) for w in weekly_tss if isinstance(w, dict)]
    monotony_proxy = None
    if len(weekly_tss_values) >= 3:
        avg_tss = sum(weekly_tss_values) / len(weekly_tss_values)
        variance = sum((v - avg_tss) ** 2 for v in weekly_tss_values) / len(weekly_tss_values)
        sd = variance ** 0.5
        monotony_proxy = round(avg_tss / sd, 2) if sd > 0 else None

    latest = metrics_compact.get("latest", {}) if isinstance(metrics_compact, dict) else {}
    sleep_latest = latest.get("sleep_hours") if isinstance(latest, dict) else None

    red_flags: list[str] = []
    if fatigue_mentions >= 2:
        red_flags.append("repeated_fatigue_language")
    if isinstance(sleep_latest, (int, float)) and sleep_latest < 6:
        red_flags.append("sleep_low_energy_availability_context_needed")
    if rpe7_count >= 2 and fatigue_mentions >= 1:
        red_flags.append("high_effort_plus_fatigue_context_needed")

    uncertainty_drivers: list[str] = []
    if not metrics_compact or metrics_compact.get("available") is False:
        uncertainty_drivers.append("readiness_metrics_missing_or_unavailable")
    if not fitness_historical or fitness_historical.get("isError"):
        uncertainty_drivers.append("fitness_historical_missing")
    if pw_hr_status.get("available_in_period_review_context") is False:
        uncertainty_drivers.append("pw_hr_not_computed_in_period_context")
    if consistency_warnings:
        uncertainty_drivers.append("derived_fact_consistency_warnings")

    if consistency_warnings:
        confidence = "low_until_reconciled"
    elif uncertainty_drivers:
        confidence = "moderate"
    else:
        confidence = "high_for_observed_facts_not_prescription"

    return {
        "purpose": "Evidence-based coaching guardrails; CODE flags domains, LLM applies judgment.",
        "pain_safety": {
            "mentions": pain_mentions,
            "doms_mentions": doms_mentions,
            "rule": (
                "Pain/DOMS are risk signals, not diagnoses. "
                "Pain altering mechanics blocks progression until clarified."
            ),
            "avoid_language": "Do not claim injury risk percentages or diagnosis from comments alone.",
        },
        "heat_environment": {
            "mentions": heat_mentions,
            "fueling_or_hydration_mentions": fueling_mentions,
            "rule": (
                "Heat/humidity can raise HR/RPE at same power; "
                "check conditions before changing fitness assumptions."
            ),
            "cali_priority": (
                "For Cali, heat is a first-class execution modifier: "
                "adjust caps/duration before judging fitness loss."
            ),
        },
        "reds_energy_availability": {
            "flags": red_flags,
            "rule": (
                "Screen for low energy availability patterns; do not prescribe aggressive weight loss "
                "or deficit during high load."
            ),
            "escalation": (
                "If persistent fatigue, recurrent pain/bone pain, illness, poor sleep or mood/libido "
                "changes appear, recommend professional evaluation."
            ),
        },
        "cp_quality": {
            "next_test_window": cycle_status.get("test_cp_ftp_scheduled"),
            "rule": (
                "Treat CP/FTP as current but auditable; test only when heat, fatigue, pain "
                "and sensor-quality blockers are absent."
            ),
            "current_thresholds_are_not_dogma": True,
        },
        "load_distribution": {
            "sessions_per_week": sessions_per_week,
            "rpe7_count": rpe7_count,
            "weekly_tss_values": weekly_tss_values[-6:],
            "monotony_proxy_from_weekly_tss": monotony_proxy,
            "rule": "Use sRPE/TSS/weekly monotony descriptively; avoid ACWR-style causal injury claims.",
        },
        "uncertainty": {
            "confidence": confidence,
            "drivers": uncertainty_drivers,
            "reporting_rule": "State decision confidence and what data would change the recommendation.",
        },
        "narrative_requirements": [
            "Distinguish physiological readiness from session/microcycle decision.",
            "Say what data is missing before making strong claims.",
            "Use CTL/ATL/TSB as load model, not absolute physiology.",
            "For pain, REDs, heat illness or medical concerns: coach conservatively and refer out when appropriate.",
        ],
    }


def _build_coaching_assessment(
    *,
    workout_summary: dict[str, Any],
    missed_summary: dict[str, Any],
    metrics_compact: dict[str, Any],
    fitness_compact: dict[str, Any],
    fitness_historical: dict[str, Any] | None,
    expanded_workouts: list[dict[str, Any]],
    period_days: int,
) -> dict[str, Any]:
    """Provide derived FACTS that the LLM would otherwise compute unreliably.

    Design principle: CODE provides facts (what IS), LLM provides judgment
    (what it MEANS and what to DO). This function computes arithmetic and
    lookups that LLMs get wrong (counting, date math, 30d windows) but does
    NOT make coaching decisions, diagnoses, or recommendations.

    What stays in code (LLMs bad at):
    - Counting RPE≥7 sessions by sport
    - 30-day max long run duration (date math)
    - Weekly run frequency counting
    - HRV deviation from average (arithmetic)
    - Cycle position from JSON timeline (date math)
    - Spurious session filtering (data quality)

    What stays in LLM (code bad at):
    - Interpreting what an RPE 7 + Feeling "Fuerte" means in context
    - Deciding whether to PROGRESS or MAINTAIN (weighing criteria)
    - Applying the 10% rule with judgment (deload context, conditions)
    - Deciding whether to follow or modify the 3:1 plan
    - Choosing which readiness signals are relevant per question type
    """
    import json as _json
    from datetime import date as date_cls
    from datetime import timedelta as td

    today = date_cls.today()
    ws = workout_summary

    # ── B: RPE≥7 sessions with Feeling context (FACTS, not diagnosis) ────
    # Provide the raw RPE + Feeling pairing so the LLM can interpret with
    # full context (heat, day variance, recovery status, etc.)
    rpe7_sessions = []
    for entry in ws.get("rpe_extremes", []):
        rpe = entry.get("rpe", 0)
        if rpe < 7:
            continue
        feeling = entry.get("feeling", {}) or {}
        rpe7_sessions.append({
            "date": entry.get("date"),
            "sport": entry.get("sport", "Unknown"),
            "rpe": rpe,
            "feeling_label": feeling.get("label", "Unknown"),
            "feeling_score_1_to_5": feeling.get("score_1_to_5", 0),
            "feeling_tp_code": feeling.get("code", None),
        })
    # Carryover: RPE≥7 within last 14 days (date math, not interpretation)
    recent_14d = today - td(days=14)
    carryover_14d = [
        s for s in rpe7_sessions
        if s["date"] and date_cls.fromisoformat(s["date"]) >= recent_14d
    ]

    # ── C: Long run 30-day max (FACT, not prescription) ──────────────────
    # The 10% rule is a guideline the LLM applies with context.
    # Code provides the 30d max; LLM decides what to do with it.
    long_runs = ws.get("long_run_progression", [])
    long_run_max = ws.get("long_run_max_duration_min", 0)
    long_run_max_date = ws.get("long_run_max_date", "")
    recent_30d = today - td(days=30)
    recent_long_runs = [
        lr for lr in long_runs
        if lr.get("date") and date_cls.fromisoformat(lr["date"]) >= recent_30d
    ]
    max_30d = max((lr.get("duration_min", 0) for lr in recent_long_runs), default=0)
    max_30d_item: dict[str, Any] = max(
        (lr for lr in recent_long_runs if lr.get("duration_min", 0) == max_30d),
        key=lambda lr: lr.get("date", ""),
        default={},
    )
    max_30d_date = max_30d_item.get("date", "")

    # ── D: Progression readiness inputs (VALUES, not pass/fail) ──────────
    # Provide raw values for each criterion. The LLM weighs them in context.
    weekly: list[dict[str, Any]] = ws.get("weekly_breakdown", []) or []

    # (1) Count weeks with 3+ runs
    weeks_with_3plus_runs = 0
    for w in weekly:
        sports_str = w.get("sports", "")
        run_count = 0
        for part in sports_str.split(", "):
            if part.startswith("Run:"):
                with suppress(ValueError, IndexError):
                    run_count = int(part.split(":")[1].split("(")[0])
        if run_count >= 3:
            weeks_with_3plus_runs += 1

    # (2) Pain signals from comments (raw flag, not diagnosis)
    # Detect pain-related keywords but exclude negations (sin dolor, no dolor, etc.)
    pain_signals = 0
    pain_keywords = ("dolor", "molestia", "lesión", "lesion", "adolorido", "pinche", "ardor")
    negation_patterns = (
        "sin dolor",
        "no dolor",
        "sin molestia",
        "no molestia",
        "sin lesión",
        "no lesión",
        "sin lesion",
    )
    for ew in expanded_workouts:
        comment = ""
        detail = ew.get("detail")
        if isinstance(detail, dict):
            raw_comments = detail.get("workout_comments", "")
            if isinstance(raw_comments, list):
                comment = " ".join(str(c) for c in raw_comments)
            elif isinstance(raw_comments, str):
                comment = raw_comments
        comment_lower = comment.lower()
        # Check for pain keywords but exclude negated mentions
        has_pain = False
        for kw in pain_keywords:
            if kw in comment_lower:
                # Check if this mention is negated
                is_negated = False
                for neg in negation_patterns:
                    if neg not in comment_lower:
                        continue
                    start_idx = comment_lower.index(neg)
                    end_idx = start_idx + len(neg) + len(kw) + 5
                    if kw in comment_lower[start_idx:end_idx]:
                        is_negated = True
                        break
                if not is_negated:
                    has_pain = True
                    break
        if has_pain:
            pain_signals += 1

    # (3) HRV deviation (arithmetic)
    mc = metrics_compact or {}
    hrv_latest = (mc.get("latest", {}) or {}).get("hrv")
    hrv_avg = (mc.get("averages", {}) or {}).get("hrv")
    hrv_deviation_pct = None
    if hrv_latest and hrv_avg and hrv_avg > 0:
        hrv_deviation_pct = round(abs(hrv_latest - hrv_avg) / hrv_avg * 100, 1)

    # (4) Strength frequency (arithmetic)
    by_sport = ws.get("by_sport", {})
    strength_sessions = by_sport.get("Strength", {}).get("sessions", 0)
    weeks_count = max(len(weekly), 1)
    strength_per_week = round(strength_sessions / weeks_count, 1)

    progression_inputs = {
        "weeks_with_3plus_runs": weeks_with_3plus_runs,
        "pain_mentions_in_comments": pain_signals,
        "pain_mentions_note": (
            "Count of workout comments containing pain-related keywords (dolor, molestia, lesión, etc.) "
            "excluding negations (sin dolor, no dolor). These are MENTIONS, not diagnosed pain events — "
            "review the comments in context to determine if any represent increasing pain that should "
            "block progression."
        ),
        "long_run_max_min": long_run_max,
        "rpe7_in_last_14d": len(carryover_14d),
        "hrv_latest": hrv_latest,
        "hrv_avg": hrv_avg,
        "hrv_deviation_pct": hrv_deviation_pct,
        "strength_per_week": strength_per_week,
        "note": (
            "These are raw inputs for progression judgment. Weigh them in context — availability, upcoming "
            "test, life stress, subjective report. The 3:1 checkpoint table in SKILL.md rule #27 is the "
            "decision framework, not a binary counter."
        ),
    }

    # ── E: Cycle 3:1 PLANNED status from JSON (context, not prescription) ─
    _block_plan_path = "/media/SSD-Storage-1/hermes-data/profiles/coach/coach-data/CURRENT_BLOCK_PLAN.json"
    cycle_status: dict[str, Any] = {}
    try:
        with open(_block_plan_path) as _f:
            bp = _json.load(_f)
        cycle_start = date_cls.fromisoformat(bp["cycle_start_date"])
        cycle_weeks = bp.get("cycle_weeks", 4)
        phases = bp.get("phases", ["Carga 1", "Carga 2", "Carga 3", "Descarga"])
        cycles_def: list[dict[str, Any]] = bp.get("cycles", []) or []

        if today < cycle_start:
            cycle_status = {"planned_status": "before_start", "note": f"Cycle starts {bp['cycle_start_date']}"}
        else:
            current_cycle: dict[str, Any] | None = None
            week_in_cycle = 0
            for cyc in cycles_def:
                cyc_start = date_cls.fromisoformat(cyc["start_date"])
                cyc_end = date_cls.fromisoformat(cyc["end_date"])
                if cyc_start <= today <= cyc_end:
                    current_cycle = cyc
                    week_in_cycle = (today - cyc_start).days // 7
                    break

            if current_cycle:
                idx = min(week_in_cycle, cycle_weeks - 1)
                cycle_num = current_cycle["number"]
                phase = phases[idx] if idx < len(phases) else f"Week {idx+1}"
                test_info = None
                for w in current_cycle.get("weeks", []):
                    if w.get("test_cp_ftp"):
                        test_info = f"{w['dates']} ({w.get('notes', 'test')})"
                        break
                if not test_info and current_cycle.get("test_cp_ftp_scheduled"):
                    test_info = current_cycle["test_cp_ftp_scheduled"]

                # Also search future cycles for the next scheduled test
                if not test_info:
                    for cyc in cycles_def:
                        if cyc["number"] <= cycle_num:
                            continue
                        for w in cyc.get("weeks", []):
                            if w.get("test_cp_ftp"):
                                test_info = f"{w['dates']} ({w.get('notes', 'test')}) — Ciclo {cyc['number']}"
                                break
                        if test_info:
                            break
                        if cyc.get("test_cp_ftp_scheduled"):
                            test_info = cyc["test_cp_ftp_scheduled"]
                            break

                # weeks_to_deload: number of weeks REMAINING until deload starts,
                # NOT counting the current week. E.g. if in week 1 of a 3:1 (4-week)
                # cycle, there are 2 more carga weeks + 1 deload = 3 weeks to deload.
                weeks_to_deload = max(cycle_weeks - idx - 1, 0)
                cycle_status = {
                    "planned_cycle": cycle_num,
                    "planned_week": idx + 1,
                    "planned_phase": phase,
                    "weeks_to_deload": weeks_to_deload,
                    "next_planned_phase": (
                        phases[(idx + 1) % cycle_weeks]
                        if idx < cycle_weeks - 1
                        else f"Ciclo {cycle_num + 1} — {phases[0]}"
                    ),
                    "test_cp_ftp_scheduled": test_info,
                    "is_plan": True,
                    "note": (
                        "This is the PLANNED cycle position from CURRENT_BLOCK_PLAN.json. You can deviate "
                        "if data justifies it — sickness, unexpected fatigue, ahead-of-schedule adaptation, "
                        "or life events may require treating a carga week as consolidation, or skipping a "
                        "deload. Use the 3:1 checkpoint table (SKILL.md rule #27) to decide."
                    ),
                }
            elif cycles_def:
                # Beyond defined cycles — extrapolate
                last_cycle = cycles_def[-1]
                last_start = date_cls.fromisoformat(last_cycle["start_date"])
                total_weeks = (today - last_start).days // 7
                cycle_num = last_cycle["number"] + total_weeks // cycle_weeks
                idx = total_weeks % cycle_weeks
                phase = phases[idx] if idx < len(phases) else f"Week {idx+1}"
                weeks_to_deload = max(cycle_weeks - idx - 1, 0)
                cycle_status = {
                    "planned_cycle": cycle_num,
                    "planned_week": idx + 1,
                    "planned_phase": phase,
                    "weeks_to_deload": weeks_to_deload,
                    "next_planned_phase": (
                        phases[(idx + 1) % cycle_weeks]
                        if idx < cycle_weeks - 1
                        else f"Ciclo {cycle_num + 1} — {phases[0]}"
                    ),
                    "is_plan": True,
                    "note": (
                        f"Extrapolated beyond defined cycles (last defined: cycle {last_cycle['number']}). "
                        "Update CURRENT_BLOCK_PLAN.json when the next cycle is confirmed."
                    ),
                }
            else:
                cycle_status = {"planned_status": "no_cycles_defined"}

            # Include bridge plan as context
            bridge = bp.get("bridge_plan", [])
            if bridge:
                cycle_status["bridge_plan"] = bridge

            # Include historical trajectory
            hist = bp.get("historical_context", {})
            if hist and fitness_historical:
                cycle_status["historical_trajectory"] = (
                    f"Peak CTL {hist.get('peak_ctl')} ({hist.get('peak_date')}) → "
                    f"nadir {hist.get('nadir_ctl')} ({hist.get('nadir_date')}) → "
                    f"current {fitness_historical.get('current_ctl', '?')}"
                )

    except (FileNotFoundError, KeyError, ValueError, _json.JSONDecodeError):
        cycle_status = {
            "planned_status": "block_plan_json_not_found",
            "note": (
                "Read CURRENT_BLOCK_PLAN.md manually at "
                "/media/SSD-Storage-1/hermes-data/profiles/coach/coach-data/CURRENT_BLOCK_PLAN.md"
            ),
        }

    # ── Global review facts: pre-formatted FACTS for block/global narratives ─
    # These are not recommendations. They prevent recurring omissions in global
    # reviews (stress, nadir, pain mentions, RPE+Feeling context, bridge/Pw:Hr).
    rpe7_line = "No RPE≥7 sessions in this period."
    rpe7_summary = ws.get("rpe7_summary", {}) if isinstance(ws, dict) else {}
    rpe7_summary_count = rpe7_summary.get("total_count") if isinstance(rpe7_summary, dict) else None
    if rpe7_sessions:
        rpe7_line = "; ".join(
            f"{s.get('date')} {s.get('sport')} RPE {s.get('rpe')} / Feeling {s.get('feeling_label')}"
            for s in rpe7_sessions
        )

    readiness_line = metrics_compact.get("readiness_snapshot") if isinstance(metrics_compact, dict) else None

    historical_line = None
    if isinstance(fitness_historical, dict) and not fitness_historical.get("isError"):
        peak = fitness_historical.get("peak_ctl")
        peak_date = fitness_historical.get("peak_date")
        nadir = fitness_historical.get("nadir_after_peak")
        nadir_date = fitness_historical.get("nadir_date")
        current = fitness_historical.get("current_ctl")
        pct = fitness_historical.get("pct_of_peak")
        historical_line = (
            f"CTL trajectory: peak {peak} ({peak_date}) → nadir {nadir} ({nadir_date}) "
            f"→ current {current} ({pct}% of peak)."
        )

    bridge_plan = cycle_status.get("bridge_plan", []) if isinstance(cycle_status, dict) else []
    bridge_plan_table = [
        {
            "phase": item.get("phase"),
            "criteria": item.get("criteria"),
            "target_date": item.get("target_date"),
        }
        for item in bridge_plan
        if isinstance(item, dict)
    ]

    pw_hr_status = {
        "available_in_period_review_context": False,
        "status": "pending_verification",
        "display": (
            "Pw:Hr/decoupling is a bridge-plan criterion but is not computed in "
            "tp_coach_period_review_context; treat it as pending verification before "
            "declaring full Rebuild → Build readiness."
        ),
    }

    pain_mentions_line = (
        f"Pain-related comments: {pain_signals} non-negated mention(s). "
        "These are mentions, not diagnosed pain events; review context and trend before using as a blocker."
    )

    bridge_markdown = "\n".join(
        f"| {item.get('phase')} | {item.get('criteria')} | {item.get('target_date')} |"
        for item in bridge_plan_table
    ) or "| N/A | Bridge plan not available | N/A |"

    consistency_warnings: list[str] = []
    if isinstance(rpe7_summary_count, int) and rpe7_summary_count != len(rpe7_sessions):
        consistency_warnings.append(
            f"rpe7_count_mismatch: workout_summary={rpe7_summary_count}, "
            f"coaching_assessment={len(rpe7_sessions)}"
        )
    metrics_available = bool(isinstance(metrics_compact, dict) and metrics_compact.get("available"))
    if metrics_available and not readiness_line:
        consistency_warnings.append("metrics_compact_available_but_readiness_line_missing")
    if fitness_historical and not historical_line:
        consistency_warnings.append("fitness_historical_available_but_historical_line_missing")
    if consistency_warnings:
        consistency_warnings.append(
            "Do not present a confident global narrative until these derived fact conflicts are reconciled."
        )

    mandatory_block = (
        "## Global review facts to include\n"
        f"- **Readiness completo:** {readiness_line or 'no disponible'}\n"
        f"- **Trayectoria CTL:** {historical_line or 'no disponible'}\n"
        f"- **Dolor/molestias:** {pain_mentions_line}\n"
        f"- **RPE≥7 + Feeling:** {rpe7_line}\n"
        f"- **Pw:Hr/decoupling:** {pw_hr_status['display']}\n"
        "\n| Fase bridge | Criterios | Fecha objetivo |\n"
        "|---|---|---|\n"
        f"{bridge_markdown}\n"
    )

    global_review_facts = {
        "purpose": "Ready-to-use FACTS for global/process/block review narratives; LLM still applies judgment.",
        "mandatory_global_review_block_markdown": mandatory_block,
        "must_include_when_available": [
            "readiness_line_all_5_signals",
            "historical_trajectory_peak_nadir_current",
            "pain_mentions_line",
            "rpe7_with_feeling_line",
            "bridge_plan_table_or_summary",
            "pw_hr_status_pending_if_not_computed",
        ],
        "consistency_warnings": consistency_warnings,
        "readiness_line_all_5_signals": readiness_line,
        "historical_trajectory_peak_nadir_current": historical_line,
        "pain_mentions_line": pain_mentions_line,
        "rpe7_with_feeling_line": rpe7_line,
        "bridge_plan_table": bridge_plan_table,
        "pw_hr_status": pw_hr_status,
        "instruction": (
            "For global/process/block reviews, include these facts explicitly if present. "
            "Do not convert them into automatic recommendations: use them as evidence for judgment."
        ),
    }

    science_guardrails = _build_science_guardrails(
        workout_summary=workout_summary,
        metrics_compact=metrics_compact,
        fitness_compact=fitness_compact,
        fitness_historical=fitness_historical,
        expanded_workouts=expanded_workouts,
        period_days=period_days,
        cycle_status=cycle_status,
        pain_mentions=pain_signals,
        pw_hr_status=pw_hr_status,
        consistency_warnings=consistency_warnings,
    )

    # ── Assemble ─────────────────────────────────────────────────────────
    return {
        "rpe7_sessions": rpe7_sessions,
        "rpe7_carryover_14d": carryover_14d,
        "long_run_30d_max_min": max_30d,
        "long_run_30d_max_date": max_30d_date,
        "long_run_block_max_min": long_run_max,
        "long_run_block_max_date": long_run_max_date,
        "progression_inputs": progression_inputs,
        "cycle_plan_status": cycle_status,
        "global_review_facts": global_review_facts,
        "science_guardrails": science_guardrails,
        "instruction": (
            "This field provides derived FACTS (counts, date math, 30d windows, "
            "HRV deviation, cycle position from JSON). Use these facts as inputs "
            "for your coaching judgment — do NOT treat them as prescriptions. "
            "You remain the decision-maker for: interpreting RPE+Feeling in context, "
            "weighing progression criteria, applying the 10% rule with judgment, "
            "and deciding whether to follow or modify the 3:1 plan."
        ),
    }


async def tp_coach_period_review_context(
    start_date: str,
    end_date: str,
    review_type: str = "period",
) -> dict[str, Any]:
    """Read-only weekly/monthly/block review context with notable-workout detail expansion.

    List views are enough for aggregation, but not for causal claims. This
    composite expands missed, key, and anomalous workouts before returning so
    weekly/monthly reports can cite comments/RPE/Feeling/private notes instead of
    guessing.

    When review_type is 'block' or 'global' (or the requested period exceeds 30
    days), a full historical fitness query from 2024-01-01 to today is
    automatically included as ``fitness_historical``. This ensures the LLM always
    has the real CTL peak and trajectory available without needing to make a
    separate tp_get_fitness call with a wide date range.
    """
    try:
        start = SingleDateInput(date=start_date).date
        end = SingleDateInput(date=end_date).date
    except (ValidationError, ValueError) as e:
        msg = format_validation_error(e) if isinstance(e, ValidationError) else str(e)
        return {"isError": True, "error_code": "VALIDATION_ERROR", "message": msg}
    if end < start:
        return {"isError": True, "error_code": "VALIDATION_ERROR", "message": "end_date must be on or after start_date"}

    start_s, end_s = start.isoformat(), end.isoformat()
    days = (end - start).days + 1

    # Determine whether to auto-inject full historical fitness context.
    # This prevents the LLM from under-querying and hallucinating peak CTL values.
    needs_historical = review_type.lower() in ("block", "global") or days > 30

    tasks = {
        "workouts": tp_get_workouts(start_s, end_s),
        "fitness": tp_get_fitness(days=days, start_date=start_s, end_date=end_s),
        "metrics": tp_get_metrics(start_s, end_s),
        "settings_summary": tp_get_athlete_settings_summary(),
        "notes": tp_list_notes(start_s, end_s),
        "availability": tp_get_availability(start_s, end_s),
    }

    # Run the standard tasks plus the historical fitness query in parallel.
    if needs_historical:
        tasks["fitness_historical"] = _compute_fitness_historical()

    results = await asyncio.gather(*tasks.values(), return_exceptions=True)
    data = dict(zip(tasks.keys(), results, strict=True))
    missing: list[str] = []
    for key, value in list(data.items()):
        if isinstance(value, Exception):
            data[key] = {"isError": True, "message": str(value)}
            missing.append(key)
        elif isinstance(value, dict) and value.get("isError"):
            missing.append(key)

    workouts = _workouts_list(data.get("workouts"))
    notes_result = data.get("notes")
    calendar_notes = notes_result.get("notes", []) if isinstance(notes_result, dict) else []
    expanded_workouts: list[dict[str, Any]] = []
    detail_expansion_ok = True
    verification: list[str] = []

    for workout in workouts:
        wid = _workout_id(workout)
        reasons = _period_expansion_reasons(workout)
        if not wid or not reasons:
            continue
        detail: Any = {"isError": True, "message": "missing_workout_id"}
        private_note: Any = {"isError": True, "message": "missing_workout_id"}
        detail, private_note = await asyncio.gather(
            tp_get_workout(workout_id=wid),
            tp_get_workout_note(workout_id=wid),
            return_exceptions=True,
        )
        if isinstance(detail, Exception):
            detail = {"isError": True, "message": str(detail)}
        if isinstance(private_note, Exception):
            private_note = {"isError": True, "message": str(private_note)}
        if isinstance(detail, dict) and not detail.get("isError"):
            verification.append(f"tp_get_workout_detail_and_private_note_called:{wid}")
        else:
            detail_expansion_ok = False

        item: dict[str, Any] = {
            "id": wid,
            "date": _workout_date(workout),
            "title": _workout_title(workout),
            "expansion_reasons": reasons,
            "list_view": workout,
            "detail": detail,
            "private_note": private_note,
        }
        if "planned_not_completed_or_missing_actuals" in reasons:
            item["reason"] = classify_missed_workout_reason(
                detail if isinstance(detail, dict) else workout,
                private_note=private_note if isinstance(private_note, dict) else {},
                calendar_notes=calendar_notes if isinstance(calendar_notes, list) else [],
            )
        else:
            item["subjective_feedback"] = normalize_subjective_feedback(detail) if isinstance(detail, dict) else None
        expanded_workouts.append(item)

    # Extract fitness_historical if it was queried (may not be present for short reviews)
    fitness_historical = data.get("fitness_historical")
    if isinstance(fitness_historical, Exception):
        fitness_historical = {"isError": True, "message": str(fitness_historical)}
        missing.append("fitness_historical")

    # ── Compact summary layer ──────────────────────────────────────────────
    # The raw expanded_workouts list can be 147K+ chars, metrics 128K+, and
    # raw workouts 58K+. The gateway truncates tool responses at ~50K, so the
    # compact summaries at the top get cut off before the LLM sees them.
    # Solution: return ONLY compact summaries by default. The raw data stays
    # available via individual tools (tp_get_workout, tp_get_metrics) for
    # drill-down. This follows the "summary + drill-down" pattern from MCP
    # community best practices (GitHub #169224, Context Mode, arxiv:2511.22729).
    workout_summary = _build_workout_summary(workouts, expanded_workouts, period_days=days)
    metrics_compact = _compact_metrics(data.get("metrics"))

    # Pre-compute missed sessions summary so the LLM doesn't have to count.
    # This prevents the recurring bug where the header says "5 omitted" but
    # the list has 6 items.
    missed_items = [
        ew for ew in expanded_workouts
        if "planned_not_completed_or_missing_actuals" in ew.get("expansion_reasons", [])
    ]
    from collections import defaultdict as _dd
    missed_by_cat: dict[str, int] = _dd(int)
    missed_list = []
    for ew in missed_items:
        reason = ew.get("reason") or {}
        cat = reason.get("category", "unknown") if isinstance(reason, dict) else "unknown"
        missed_by_cat[cat] += 1
        missed_list.append({
            "date": ew.get("date"),
            "title": ew.get("title"),
            "category": cat,
            "evidence": reason.get("evidence", "") if isinstance(reason, dict) else "",
        })
    missed_summary = {
        "total_count": len(missed_items),
        "by_category": dict(missed_by_cat),
        "sessions": missed_list,
        "coaching_note": (
            f"There are {len(missed_items)} missed/planned-not-completed sessions. "
            f"By category: {', '.join(f'{k}: {v}' for k, v in sorted(missed_by_cat.items()))}. "
            f"Report the TOTAL count as {len(missed_items)} — do NOT count manually."
        ) if missed_items else "No missed sessions in this period.",
    }

    # Build compact versions of notes and availability too
    notes_compact = []
    if isinstance(data.get("notes"), dict):
        for note in (data.get("notes", {}).get("notes", []))[:10]:  # Last 10 notes
            if isinstance(note, dict):
                notes_compact.append({
                    "date": note.get("date") or note.get("workoutDay", ""),
                    "type": note.get("type", ""),
                    "text": (note.get("note") or note.get("text") or "")[:200],
                })

    # ── Coaching assessment (pre-computed derived analysis) ─────────────
    # This replaces rules #19, #26, #27, #33, #37, #41, #49 with code-computed
    # values that the LLM reads directly instead of having to interpret.
    coaching_assessment = _build_coaching_assessment(
        workout_summary=workout_summary,
        missed_summary=missed_summary,
        metrics_compact=metrics_compact,
        fitness_compact=_compact_fitness(data.get("fitness")),
        fitness_historical=fitness_historical if isinstance(fitness_historical, dict) else None,
        expanded_workouts=expanded_workouts,
        period_days=days,
    )

    return {
        "review_type": review_type,
        "period": {"start": start_s, "end": end_s, "days": days},
        # ── Mandatory global-review facts first (maximum LLM visibility) ──
        "global_review_facts": coaching_assessment.get("global_review_facts"),
        # ── Compact summaries (always visible to LLM, ~12K total) ──
        "workout_summary": workout_summary,
        "missed_summary": missed_summary,
        "coaching_assessment": coaching_assessment,
        "metrics_compact": metrics_compact,
        "fitness_compact": _compact_fitness(data.get("fitness")),
        "fitness_historical": fitness_historical,
        "settings_summary": data.get("settings_summary"),
        "notes_recent": notes_compact,
        "availability": data.get("availability"),
        # ── Expanded workout highlights (compact: only key/missed sessions) ──
        "expanded_workout_highlights": [
            {
                "id": ew.get("id"),
                "date": ew.get("date"),
                "title": ew.get("title"),
                "expansion_reasons": ew.get("expansion_reasons"),
                "subjective_feedback": ew.get("subjective_feedback"),
                "reason": ew.get("reason"),
            }
            for ew in expanded_workouts
        ],
        # ── Guardrails ──
        "decision_guardrails": {
            "must_expand_notable_workouts_before_causal_claims": True,
            "notable_workout_policy": (
                "expand missed/planned-not-completed, key sessions, and plan-vs-actual anomalies before "
                "interpreting causes or adaptation"
            ),
            "can_make_period_causal_claims": detail_expansion_ok,
            "must_not_compensate_missed_tss_automatically": True,
            "must_use_fitness_historical_for_peak_ctl": bool(
                fitness_historical
                and not (isinstance(fitness_historical, dict) and fitness_historical.get("isError"))
            ),
            "fitness_historical_rule": (
                "When fitness_historical is present and not an error, its peak_ctl, peak_date, "
                "pct_of_peak, and coaching_instruction are the GROUND TRUTH for any historical "
                "comparison. Do NOT cite any other CTL peak value from memory, prior sessions, "
                "or the narrow-period fitness field. The fitness_historical values come from "
                "a 2024-01-01 to today query and override all other sources."
            ),
            "workout_summary_rule": (
                "The workout_summary field contains pre-computed highlights: long run progression, "
                "weekly TSS with dates and sport breakdown, new maximums, and notable sessions. "
                "USE workout_summary for narrative — do NOT re-derive from expanded_workout_highlights. "
                "If you need full detail on a specific workout (laps, power data, comments), "
                "call tp_get_workout with the id from expanded_workout_highlights."
            ),
        },
        "verification": verification,
        "missing_or_error_sources": missing,
        "safety": {"read_only": True, "writes_allowed": False},
    }


async def tp_coach_readiness_snapshot(
    date_str: str | None = None,
    baseline_days: int = 28,
) -> dict[str, Any]:
    """Read-only composite readiness snapshot for Fitnessbot daily decisions."""
    try:
        ref = SingleDateInput(date=date_str).date if date_str else date.today()
    except (ValidationError, ValueError) as e:
        msg = format_validation_error(e) if isinstance(e, ValidationError) else str(e)
        return {"isError": True, "error_code": "VALIDATION_ERROR", "message": msg}

    start = ref - timedelta(days=max(1, baseline_days - 1))
    metrics_result, fitness_result = await asyncio.gather(
        tp_get_metrics(start.isoformat(), ref.isoformat()),
        tp_get_fitness(days=baseline_days, start_date=start.isoformat(), end_date=ref.isoformat()),
    )
    metrics_payload = metrics_result if isinstance(metrics_result, dict) else {}
    latest, freshness = _metric_dict_for_date(metrics_payload, ref.isoformat())
    metrics = (metrics_result or {}).get("metrics", []) if isinstance(metrics_result, dict) else []

    def avg_metric(name: str) -> float | None:
        vals = []
        for item in metrics:
            if isinstance(item, dict):
                v = metric_value(_latest_metric_dict({"metrics": [item]}), name)
                if v is not None:
                    vals.append(v)
        return round(sum(vals) / len(vals), 1) if vals else None

    baselines = {"hrv": avg_metric("hrv"), "pulse": avg_metric("pulse")}
    daily = (fitness_result or {}).get("daily_data", []) if isinstance(fitness_result, dict) else []
    fitness_same_day = [
        item
        for item in daily
        if isinstance(item, dict)
        and str(item.get("date") or item.get("day") or item.get("workoutDay") or "")[:10] == ref.isoformat()
    ]
    fitness_latest = fitness_same_day[-1] if fitness_same_day else (daily[-1] if daily else {})
    fitness_source_date = str(
        (fitness_latest or {}).get("date")
        or (fitness_latest or {}).get("day")
        or (fitness_latest or {}).get("workoutDay")
        or ""
    )[:10]
    fitness_freshness = {
        "requested_date": ref.isoformat(),
        "source_date": fitness_source_date or None,
        "is_current_day": bool(fitness_same_day),
        "decision_rule": (
            "do_not_use_stale_fitness_as_today"
            if not fitness_same_day
            else "ok_to_use_for_today_load_trend"
        ),
    }
    classification = classify_readiness_snapshot(latest, baselines, fitness_latest)
    physiological_classification = classify_readiness_snapshot(latest, baselines, {})
    session_classification = {
        **classification,
        "interpretation": (
            "This is the session/microcycle decision light after combining physiological readiness "
            "with load trend/TSB; it can be more conservative than physiological readiness."
        ),
    }
    readiness_layers = {
        "physiological_readiness": {
            **physiological_classification,
            "interpretation": "HRV/sleep/pulse-style systemic/autonomic readiness before load/TSB context.",
        },
        "session_decision": session_classification,
        "reporting_rule": (
            "If physiological readiness is green but session decision is yellow/orange, say so plainly: "
            "readiness physiology is good, but today remains conservative because of recent load, TSB, "
            "planned recovery objective, or legs/impact."
        ),
    }
    body_battery = _body_battery_interpretation(latest)
    decision_guardrails = {
        "readiness_metrics_are_current_day": freshness.get("is_current_day") is True,
        "must_not_use_stale_readiness_as_today": freshness.get("is_current_day") is not True,
        "affected_readiness_signals_to_name": ["HRV", "sleep", "pulse/RHR-proxy", "Body Battery", "stress"],
        "explain_freshness_status_plainly": True,
        "if_stale_or_missing": (
            "Say same-day overnight metrics are not yet fresh/confirmed; name the affected signals "
            "(HRV, sleep, pulse/RHR-proxy, Body Battery, stress) and do not present fallback metrics as "
            "today's readiness. If metric_freshness has source_date/requested_date, mention them compactly. "
            "Base the decision conservatively on load, yesterday's session, availability, planned workout, "
            "and subjective check-in until current metrics arrive."
        ),
        "body_battery_rule": (
            "Body Battery max value IS the wake/end-of-sleep value by Garmin's design (Garmin official: "
            "'Body Battery is fullest in the morning when you wake up'; TP syncs 'the highest and lowest "
            "value for each day'). Report it as: '{max} al despertar (mínimo del día: {min}, promedio: {avg})'. "
            "Use the wake value as the primary readiness number: >=80 strong recovery support, 50-79 moderate, "
            "<50 low. Frame it as recovery support, not permission to add load."
        ),
        "load_language_rule": (
            "Use calibrated load language: TSB around -10 after a long run is expected functional fatigue, "
            "not an alarm; it supports recovery movement, not additional stimulus."
        ),
        "combined_readiness_load_rule": (
            "When HRV/sleep/pulse/Body Battery look good but recent TSS/duration or TSB show load, say: "
            "systemic/autonomic recovery is good, but the limiter is mechanical legs/impact and microcycle "
            "load. Use the combined read: good physiology + elevated load = execute easy, no extra."
        ),
        "good_readiness_does_not_add_load_rule": (
            "Say explicitly when relevant: good readiness authorizes executing the recovery session; it does not "
            "authorize extending, chasing pace, or adding training load."
        ),
        "recovery_execution_fallback_rule": (
            "For recovery sessions, include simple limiters: if RPE drifts to 3/10 sustained, or if HR/breathing "
            "feels high for the target power, drop 10-15 W or walk; if DOMS/pain changes mechanics, switch "
            "to walking/mobility."
        ),
        "brevity_rule": (
            "Keep routine Daily Briefs compact: decision, 3-5 evidence bullets, execution caps, and report prompt. "
            "Only expand when data quality, safety, or plan changes need explanation."
        ),
    }
    if freshness.get("is_current_day") is not True:
        classification = {
            **classification,
            "traffic_light": "unknown_stale_metrics",
            "decision_bias": "conservative_until_same_day_metrics_arrive",
            "flags": [*classification.get("flags", []), "readiness_metrics_not_current_day"],
        }
        readiness_layers["session_decision"] = {
            **classification,
            "interpretation": "Conservative session decision because same-day readiness metrics are stale/missing.",
        }
    return {
        "date": ref.isoformat(),
        "latest_metrics": latest,
        "metric_freshness": freshness,
        "baselines": baselines,
        "fitness": fitness_latest,
        "fitness_freshness": fitness_freshness,
        "body_battery_interpretation": body_battery,
        "readiness_layers": readiness_layers,
        "classification": classification,
        "decision_guardrails": decision_guardrails,
        "missing_or_error_sources": [
            key for key, result in {"metrics": metrics_result, "fitness": fitness_result}.items()
            if isinstance(result, dict) and result.get("isError")
        ],
        "safety": {"read_only": True, "writes_allowed": False},
    }


async def tp_coach_workout_compliance_v2(workout_id: str) -> dict[str, Any]:
    """Read-only composite post-workout compliance context."""
    from tp_mcp.tools.analyze import tp_analyze_workout

    detail: Any
    analysis: Any
    detail, analysis = await asyncio.gather(
        tp_get_workout(workout_id=workout_id),
        tp_analyze_workout(workout_id=workout_id),
        return_exceptions=True,
    )
    if isinstance(detail, Exception):
        detail = {"isError": True, "message": str(detail)}
    if isinstance(analysis, Exception):
        analysis = {"isError": True, "message": str(analysis)}
    workouts = [detail] if isinstance(detail, dict) and not detail.get("isError") else []
    return {
        "workout_id": workout_id,
        "detail": detail,
        "analysis": analysis,
        "subjective_feedback": normalize_subjective_feedback(detail)
        if isinstance(detail, dict) and not detail.get("isError")
        else None,
        "feedback_summary": summarize_feedback_patterns(workouts),
        "compliance_axes": [
            "duration", "load_tss", "intensity", "execution_laps_or_structure",
            "subjective_rpe_feeling_comments", "microcycle_fit",
        ],
        "safety": {"read_only": True, "writes_allowed": False},
    }


async def tp_coach_plan_guardrails(
    plan: dict[str, Any],
    availability_by_date: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate a neutral weekly plan before any TrainingPeaks write."""
    normalized_avail: dict[str, dict[str, Any]] = {}
    for key, value in (availability_by_date or {}).items():
        normalized_avail[str(key)] = value if isinstance(value, dict) else {"available": bool(value)}
    result = validate_week_plan_guardrails(plan, normalized_avail)
    result["safety"] = {
        "read_only": True,
        "writes_allowed": False,
        "message": "Validation only. Passing guardrails is not approval to write TrainingPeaks.",
    }
    return result
