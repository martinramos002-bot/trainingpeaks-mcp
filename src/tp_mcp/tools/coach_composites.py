"""Coach-oriented composite tools for Fitnessbot.

Read-only wrappers around existing TrainingPeaks tools plus pure guardrail helpers.
Do not add create/update/delete behavior here.
"""

from __future__ import annotations

import asyncio
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
            "If the source only provides a range/list, do not call any value the wake/end-of-sleep value "
            "unless that timestamp is explicitly confirmed."
        ),
    }
    if isinstance(raw, list):
        nums = [metric_value({"value": v}, "value") for v in raw]
        nums = [v for v in nums if v is not None]
        if len(nums) >= 3:
            interpretation.update({
                "format": "array_min_max_avg",
                "min": round(nums[0], 1),
                "max": round(nums[1], 1),
                "avg": round(nums[2], 1),
                "display_guidance": (
                    f"Body Battery: rango TP aprox. {nums[0]:.0f}→{nums[1]:.0f}, promedio ~{nums[2]:.0f}; "
                    "señal de recuperación, no permiso para sumar carga."
                ),
            })
        elif nums:
            interpretation.update({"format": "array_values", "values": [round(v, 1) for v in nums]})
    else:
        value = metric_value({"body_battery": raw}, "body_battery")
        if value is not None:
            interpretation.update({
                "format": "single_value",
                "value": round(value, 1),
                "display_guidance": (
                    f"Body Battery: {value:.0f}; señal de recuperación, no permiso para sumar carga."
                ),
            })
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
                "When useful, separate physiological readiness light from session/microcycle decision light: "
                "e.g. readiness fisiológica verde, decisión de sesión amarilla por carga/impacto."
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
    return {
        "date": ref_s,
        "week": {"start": start.isoformat(), "end": end.isoformat()},
        "today": today_workouts,
        "yesterday": expanded_yesterday,
        "readiness": data.get("readiness"),
        "week_context": data.get("week_context"),
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
    tasks = {
        "workouts": tp_get_workouts(start_s, end_s),
        "fitness": tp_get_fitness(days=days, start_date=start_s, end_date=end_s),
        "metrics": tp_get_metrics(start_s, end_s),
        "settings_summary": tp_get_athlete_settings_summary(),
        "notes": tp_list_notes(start_s, end_s),
        "availability": tp_get_availability(start_s, end_s),
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

    return {
        "review_type": review_type,
        "period": {"start": start_s, "end": end_s, "days": days},
        "workouts": data.get("workouts"),
        "fitness": data.get("fitness"),
        "metrics": data.get("metrics"),
        "settings_summary": data.get("settings_summary"),
        "notes": data.get("notes"),
        "availability": data.get("availability"),
        "expanded_workouts": expanded_workouts,
        "decision_guardrails": {
            "must_expand_notable_workouts_before_causal_claims": True,
            "notable_workout_policy": (
                "expand missed/planned-not-completed, key sessions, and plan-vs-actual anomalies before "
                "interpreting causes or adaptation"
            ),
            "can_make_period_causal_claims": detail_expansion_ok,
            "must_not_compensate_missed_tss_automatically": True,
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
            "If Body Battery is present as a range/timeline, show the range/average explicitly (e.g. "
            "TP range 63→100, avg ~90) and frame it as recovery support, not permission to add load. "
            "Distinguish sleep-start/pre-sleep, overnight recovery, and wake/end-of-sleep value; never treat "
            "pre-sleep as the wake value unless the source timestamp confirms it."
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
