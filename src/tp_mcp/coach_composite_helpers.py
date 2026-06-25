"""Pure coach-composite helpers for Fitnessbot.

No TrainingPeaks network calls and no heavy MCP tool imports. Keep this module
pure so it is safe to unit test quickly.
"""

from __future__ import annotations

from collections import Counter
from datetime import date, timedelta
from typing import Any

FEELING_LABELS = {
    # TP/Garmin uses odd internal codes with lower=better and higher=worse.
    # Martín confirmed a workout marked visually as "Débil" came through as 7.
    1: "Muy fuerte",
    3: "Fuerte",
    5: "Normal",
    7: "Débil",
    9: "Muy débil",
}
FEELING_SCORES_1_TO_5 = {1: 5, 3: 4, 5: 3, 7: 2, 9: 1}


def normalize_feeling(value: Any) -> dict[str, Any]:
    """Normalize TP/Garmin odd-code feeling into Martín's reporting format."""
    try:
        code = int(value) if value is not None else None
    except (TypeError, ValueError):
        code = None
    if code not in FEELING_LABELS:
        return {
            "code": code,
            "label": None,
            "score_1_to_5": None,
            "display": None,
            "warning": "unknown_or_missing_feeling_code" if value is not None else "missing_feeling",
        }
    score = FEELING_SCORES_1_TO_5[code]
    label = FEELING_LABELS[code]
    return {
        "code": code,
        "label": label,
        "score_1_to_5": score,
        "display": f"Feeling {label} ({score}/5; TP code {code})",
        "warning": "feeling_scale_is_inverse_higher_is_worse" if code in {1, 3, 7, 9} else None,
    }


def normalize_subjective_feedback(workout: dict[str, Any]) -> dict[str, Any]:
    """Normalize RPE + feeling without mixing effort cost and subjective quality."""
    rpe = as_float(workout.get("rpe"))
    return {
        "rpe": None if rpe is None else round(rpe, 1),
        "rpe_scale": "/10",
        "feeling": normalize_feeling(workout.get("feeling")),
        "interpretation_rule": (
            "RPE is effort/cost; Feeling is subjective quality/readiness. "
            "TP/Garmin feeling codes are inverse: 1/3/5/7/9 = Muy fuerte/Fuerte/Normal/Débil/Muy débil. "
            "Do not treat code 7 as positive; Martín confirmed it means Débil."
        ),
    }


def week_bounds(ref_date: date) -> tuple[date, date]:
    """Return Monday-Sunday bounds for the week containing ref_date."""
    monday = ref_date - timedelta(days=ref_date.weekday())
    return monday, monday + timedelta(days=6)


def _collect_text(value: Any) -> list[str]:
    """Extract human text from TP comments/notes without trusting structure."""
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    if isinstance(value, dict):
        texts: list[str] = []
        for key in ("comment", "newComment", "note", "description", "title", "text"):
            texts.extend(_collect_text(value.get(key)))
        return texts
    if isinstance(value, list):
        texts = []
        for item in value:
            texts.extend(_collect_text(item))
        return texts
    return []


def classify_missed_workout_reason(
    workout: dict[str, Any],
    *,
    private_note: dict[str, Any] | None = None,
    calendar_notes: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Classify why a planned workout was missed from explicit TP text evidence.

    This deliberately prefers athlete workout comments over inference from a
    workout title such as "recuperación". A missed workout stays ambiguous until
    a comment/note/calendar entry gives a reason.
    """
    text_parts: list[str] = []
    text_parts.extend(_collect_text(workout.get("workout_comments")))
    text_parts.extend(_collect_text(workout.get("new_comment")))
    text_parts.extend(_collect_text(private_note or {}))
    for note in calendar_notes or []:
        text_parts.extend(_collect_text(note))
    evidence = " | ".join(dict.fromkeys(t for t in text_parts if t))
    lowered = evidence.lower()

    category = "unknown"
    confidence = "low"
    time_logistics_terms = (
        "no tuve tiempo",
        "sin tiempo",
        "falta de tiempo",
        "por tiempo",
        "logística",
        "logistica",
        "trabajo",
        "agenda",
        "reunión",
        "reunion",
    )
    fatigue_recovery_terms = (
        "fatiga",
        "cansado",
        "cansancio",
        "agotado",
        "piernas pesadas",
        "doms",
        "recuperar",
        "recuperación",
        "recuperacion",
    )

    if any(k in lowered for k in time_logistics_terms):
        category = "time_logistics"
        confidence = "high"
    elif any(k in lowered for k in fatigue_recovery_terms):
        category = "fatigue_recovery"
        confidence = "high"
    elif any(k in lowered for k in ("dolor", "molestia", "lesión", "lesion")):
        category = "pain_injury"
        confidence = "high"
    elif any(k in lowered for k in ("enfermo", "fiebre", "resfr", "gripe", "malestar")):
        category = "illness"
        confidence = "high"
    elif any(k in lowered for k in ("sync", "sincron", "garmin", "stryd", "archivo", "no subió", "no subio")):
        category = "sync_failure"
        confidence = "medium"
    elif any(k in lowered for k in ("caminata", "movilidad", "hice bici", "otra actividad")):
        category = "alternate_activity"
        confidence = "medium"

    return {
        "category": category,
        "confidence": confidence,
        "evidence": evidence,
        "decision_rule": "do_not_compensate_missed_tss",
    }


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, list):
        nums = [as_float(v) for v in value]
        nums = [v for v in nums if v is not None]
        return nums[-1] if nums else None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def metric_value(metrics: dict[str, Any], *names: str) -> float | None:
    lowered = {str(k).lower(): v for k, v in metrics.items()}
    for name in names:
        if name in metrics:
            return as_float(metrics[name])
        key = name.lower()
        if key in lowered:
            return as_float(lowered[key])
    return None


def classify_readiness_snapshot(
    latest_metrics: dict[str, Any] | None,
    baselines: dict[str, Any] | None = None,
    fitness: dict[str, Any] | None = None,
    subjective_flags: list[str] | None = None,
) -> dict[str, Any]:
    """Classify readiness as green/yellow/orange/red from common TP signals."""
    latest_metrics = latest_metrics or {}
    baselines = baselines or {}
    fitness = fitness or {}
    subjective_flags = subjective_flags or []

    flags: list[str] = []
    severity = 0

    sleep = metric_value(latest_metrics, "sleep_hours", "sleep", "Sleep")
    if sleep is not None:
        if sleep < 5.5:
            flags.append("sleep_low")
            severity += 2
        elif sleep < 6.5:
            flags.append("sleep_borderline")
            severity += 1

    hrv = metric_value(latest_metrics, "hrv", "HRV")
    hrv_base = metric_value(baselines, "hrv", "HRV")
    if hrv is not None and hrv_base:
        if hrv < hrv_base * 0.8:
            flags.append("hrv_low_vs_baseline")
            severity += 2
        elif hrv < hrv_base * 0.9:
            flags.append("hrv_borderline_vs_baseline")
            severity += 1

    pulse = metric_value(latest_metrics, "pulse", "rhr", "resting_hr", "Pulse")
    pulse_base = metric_value(baselines, "pulse", "rhr", "resting_hr", "Pulse")
    if pulse is not None and pulse_base:
        if pulse >= pulse_base + 8:
            flags.append("pulse_high_vs_baseline")
            severity += 2
        elif pulse >= pulse_base + 5:
            flags.append("pulse_borderline_vs_baseline")
            severity += 1

    tsb = metric_value(fitness, "tsb", "form")
    if tsb is not None:
        if tsb <= -20:
            flags.append("tsb_very_negative")
            severity += 2
        elif tsb <= -10:
            flags.append("tsb_negative")
            severity += 1

    # Body Battery wake value (max of array) as readiness support
    bb_raw = latest_metrics.get("body_battery")
    if bb_raw is None:
        bb_raw = latest_metrics.get("Body Battery")
    if bb_raw is not None:
        bb_wake: float | None = None
        if isinstance(bb_raw, list):
            bb_nums = [metric_value({"value": v}, "value") for v in bb_raw]
            bb_nums = [v for v in bb_nums if v is not None]
            if len(bb_nums) >= 2:
                bb_wake = bb_nums[1]  # max = wake value
            elif bb_nums:
                bb_wake = bb_nums[0]
        else:
            bb_wake = metric_value({"body_battery": bb_raw}, "body_battery")
        if bb_wake is not None:
            if bb_wake < 30:
                flags.append("body_battery_low_wake")
                severity += 2
            elif bb_wake < 50:
                flags.append("body_battery_borderline_wake")
                severity += 1

    normalized_subjective = {f.lower().strip() for f in subjective_flags}
    if normalized_subjective & {"illness", "sick", "fever", "red", "acute_pain"}:
        flags.append("subjective_red_flag")
        severity += 4
    if normalized_subjective & {"fatigue", "pain", "doms", "stress", "heat"}:
        flags.append("subjective_caution")
        severity += 1

    if severity >= 4 or "subjective_red_flag" in flags:
        light = "red" if "subjective_red_flag" in flags else "orange"
    elif severity >= 2:
        light = "orange"
    elif severity == 1:
        light = "yellow"
    else:
        light = "green"

    decision_bias = {
        "green": "execute_as_planned",
        "yellow": "cap_or_trim_if_needed",
        "orange": "reduce_or_recovery",
        "red": "rest_or_medical_guardrail",
    }[light]

    return {
        "traffic_light": light,
        "flags": flags,
        "severity": severity,
        "decision_bias": decision_bias,
    }


def _planned_distance_km(item: dict[str, Any]) -> float:
    """Extract a planned distance in km from a plan day item, tolerating naming variants."""
    for key in ("distance_km", "distance_planned_km", "distance_min", "distance_planned", "planned_distance_km"):
        value = as_float(item.get(key))
        if value:
            return value
    return 0.0


def validate_week_plan_guardrails(
    plan: dict[str, Any],
    availability_by_date: dict[str, dict[str, Any]] | None = None,
    *,
    long_run_history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Validate Fitnessbot weekly-plan guardrails before any TP write.

    ``long_run_history`` is an optional list of recent long-run sessions used to
    enforce the SOUL.md 10% rule: no planned session may exceed 110% of the
    longest long run in the last 30 days. Each entry should carry a ``date`` (or
    ``workoutDay``) and a distance in km under any of the keys
    ``distance_km``/``distance_actual_km``/``distance_planned_km``. Entries
    older than 30 days from today are ignored. When the list is empty or not
    provided, a soft warning is emitted instead of a violation.
    """
    availability_by_date = availability_by_date or {}
    days = plan.get("days") or plan.get("workouts") or []
    violations: list[str] = []
    warnings: list[str] = []

    priority = plan.get("priority") or plan.get("primary_priority")
    if isinstance(priority, (list, tuple, set)) and len(priority) != 1:
        violations.append("multiple_primary_priorities")
    elif not priority:
        warnings.append("primary_priority_missing")

    # --- 10% long-run rule setup -------------------------------------------
    recent_long_run_max_km = 0.0
    long_run_history = long_run_history or []
    today = date.today()
    thirty_days_ago = today - timedelta(days=30)
    for entry in long_run_history:
        entry_day_raw = str(entry.get("date") or entry.get("workoutDay") or "")[:10]
        try:
            entry_day = date.fromisoformat(entry_day_raw) if entry_day_raw else None
        except ValueError:
            entry_day = None
        if entry_day is not None and entry_day < thirty_days_ago:
            continue
        dist = 0.0
        for key in ("distance_km", "distance_actual_km", "distance_planned_km"):
            value = as_float(entry.get(key))
            if value:
                dist = value
                break
        if dist > recent_long_run_max_km:
            recent_long_run_max_km = dist

    long_run_rule_cap_km = recent_long_run_max_km * 1.10 if recent_long_run_max_km > 0 else None
    long_run_rule_active = long_run_rule_cap_km is not None

    run_quality = 0
    has_long_run = False
    has_strength_or_mobility = False
    total_duration = 0.0
    total_tss = 0.0

    for item in days:
        day = str(item.get("date") or item.get("workoutDay") or "")[:10]
        sport = str(item.get("sport") or item.get("workout_type") or item.get("type") or "").lower()
        intensity = str(item.get("intensity") or item.get("theme") or item.get("title") or "").lower()
        duration = as_float(item.get("duration_min") or item.get("duration_minutes")) or 0.0
        tss = as_float(item.get("tss") or item.get("tss_planned")) or 0.0
        total_duration += duration
        total_tss += tss

        avail = availability_by_date.get(day)
        if avail and avail.get("available") is False and sport not in {"", "rest", "off"}:
            violations.append(f"training_on_unavailable_day:{day}")

        if sport in {"strength", "mobility", "yoga", "other"} or "movilidad" in intensity or "fuerza" in intensity:
            has_strength_or_mobility = True

        if sport == "run" or "run" in sport or "running" in sport:
            if any(k in intensity for k in ("tempo", "threshold", "vo2", "hill", "interval", "umbral")):
                run_quality += 1
            if any(k in intensity for k in ("long", "tirada", "larga")) or duration >= 75:
                has_long_run = True

            # 10% long-run rule: check every run session's planned distance.
            if long_run_rule_active:
                planned_km = _planned_distance_km(item)
                if planned_km > 0 and planned_km > long_run_rule_cap_km:  # type: ignore[operator]
                    violations.append(
                        f"long_run_exceeds_10pct_rule:{day}:{round(planned_km, 1)}km>"
                        f"{round(long_run_rule_cap_km, 1)}km"
                    )

    if run_quality >= 2:
        violations.append("too_many_run_quality_sessions")
    if has_long_run and run_quality >= 1 and len(days) < 5:
        warnings.append("long_run_plus_quality_in_short_week")
    if not has_strength_or_mobility:
        warnings.append("strength_or_mobility_missing")
    if total_duration and total_tss and total_tss / max(total_duration / 60.0, 0.01) > 90:
        warnings.append("high_tss_density_check_thresholds")

    if not long_run_rule_active:
        warnings.append("long_run_10pct_rule_not_evaluated_no_history")

    return {
        "ok": not violations,
        "violations": violations,
        "warnings": warnings,
        "summary": {
            "planned_items": len(days),
            "total_duration_min": round(total_duration, 1),
            "total_tss": round(total_tss, 1),
            "has_strength_or_mobility": has_strength_or_mobility,
            "run_quality_sessions": run_quality,
            "has_long_run": has_long_run,
            "long_run_10pct_rule": {
                "active": long_run_rule_active,
                "recent_long_run_max_km": round(recent_long_run_max_km, 2) if recent_long_run_max_km else None,
                "cap_km": round(long_run_rule_cap_km, 2) if long_run_rule_cap_km else None,
            },
        },
    }


def summarize_feedback_patterns(workouts: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize RPE/Feeling/comments patterns from workout dicts."""
    rpes: list[float] = []
    feeling_labels: list[str] = []
    flags: Counter[str] = Counter()

    for workout in workouts:
        rpe = as_float(workout.get("rpe"))
        if rpe is not None:
            rpes.append(rpe)
        feeling = workout.get("feeling")
        try:
            feeling_code = int(feeling) if feeling is not None else None
        except (TypeError, ValueError):
            feeling_code = None
        if feeling_code in FEELING_LABELS:
            feeling_labels.append(FEELING_LABELS[feeling_code])

        comment_parts = [
            str(workout.get("workout_comments") or ""),
            str(workout.get("athlete_comment") or ""),
            str(workout.get("comments") or ""),
        ]
        comments = " ".join(comment_parts).lower()
        # Keep negated symptom reports ("sin dolor", "no pain") from counting
        # as positive risk flags. This is intentionally conservative/simple;
        # full NLP is not needed for weekly triage.
        pain_text = comments
        for negated in (
            "sin dolor",
            "sin molestias",
            "no dolor",
            "no pain",
            "without pain",
            "no soreness",
        ):
            pain_text = pain_text.replace(negated, " ")
        if any(k in comments for k in ("calor", "heat", "hot")):
            flags["heat"] += 1
        if any(k in pain_text for k in ("dolor", "pain", "molestia")):
            flags["pain"] += 1
        if any(k in comments for k in ("doms", "agujeta", "agujetas", "soreness")):
            flags["doms"] += 1
        if any(k in comments for k in ("fatiga", "fatigue", "cansado")):
            flags["fatigue"] += 1
        if any(k in comments for k in ("gi", "estómago", "gastro", "fuel", "gel")):
            flags["fueling_gi"] += 1

    avg_rpe = round(sum(rpes) / len(rpes), 1) if rpes else None
    return {
        "count": len(workouts),
        "avg_rpe": avg_rpe,
        "feeling_labels": feeling_labels,
        "flags": dict(flags),
    }
