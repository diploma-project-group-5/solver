"""Tests of bench/analyze.py on synthetic data (no bench results are read)."""

import pytest

import analyze as a

SNAP = {
    "kind": "main", "staff": 2, "capacity": {"passport": 40},
    "classes": [{"id": "c1", "students": 10, "priority": False}, {"id": "c2", "students": 20, "priority": False}],
    "subjects": [{"id": "m", "priority_area": True, "no_shelter": False},
                 {"id": "art", "priority_area": False, "no_shelter": False}],
    "teachers": [{"id": "t1", "unavailable_days": [], "weekly_limit": 20}],
    "rows": [{"class_id": "c1", "subject_id": "m", "teacher_id": "t1", "hours": 2},
             {"class_id": "c2", "subject_id": "m", "teacher_id": "t1", "hours": 1},
             {"class_id": "c2", "subject_id": "art", "teacher_id": "t1", "hours": 1}],
}
TT = {
    "lessons": [{"class_id": "c1", "day": 1, "lesson_no": 1, "subject_id": "m", "teacher_id": "t1"},
                {"class_id": "c1", "day": 2, "lesson_no": 1, "subject_id": "m", "teacher_id": "t1"},
                {"class_id": "c2", "day": 1, "lesson_no": 2, "subject_id": "m", "teacher_id": "t1"},
                {"class_id": "c2", "day": 1, "lesson_no": 3, "subject_id": "art", "teacher_id": "t1"}],
    "day_formats": [{"class_id": "c1", "day": 1, "in_person": True}, {"class_id": "c1", "day": 2, "in_person": False},
                    {"class_id": "c2", "day": 1, "in_person": True}, {"class_id": "c2", "day": 2, "in_person": False}],
}


def test_primary_counts_priority_in_person_weighted_by_pupils():
    assert a.primary_of(SNAP, TT) == (10 + 20, 10 * 2 + 20 * 1)


def test_alert_no_shelter_priority_lessons_are_not_counted():
    snap = {**SNAP, "kind": "alert", "capacity": {"passport": 40, "seats": 25},
            "subjects": [{"id": "m", "priority_area": True, "no_shelter": True}, SNAP["subjects"][1]]}
    assert a.primary_of(snap, TT) == (0, 0)


def test_inputs_delta_and_tight():
    inp = a.inputs_of(SNAP)  # 30 pupils + 1 teacher + 2 staff = 33 <= 40 -> not tight
    assert (inp.pmax, inp.priority_pupils, inp.tight) == (40, 30, False)
    assert inp.delta == pytest.approx(30 / 40)
    assert a.inputs_of({**SNAP, "capacity": {"passport": 32}}).tight


@pytest.mark.parametrize(("row", "line", "expected"), [
    ({"case_id": "x", "status": "INFEASIBLE", "variant": "joint"}, None, (0.0, "no_timetable")),
    ({"case_id": "x", "status": "FEASIBLE", "validator_status": "REJECTED", "variant": "rotation"}, None, (0.0, "rejected")),
    ({"case_id": "x", "status": "OPTIMAL", "validator_status": "ACCEPTED", "variant": "rotation"},
     {"input": {"snapshot": SNAP, "timetable": TT}}, (30 / 40, "accepted")),
])
def test_run_share_rules(row, line, expected):
    assert a.run_share(row, line) == pytest.approx(expected)


@pytest.mark.parametrize("row", [
    {"case_id": "x", "status": "CRASH", "variant": "joint"},
    {"case_id": "x", "status": "FEASIBLE", "validator_status": "", "variant": "joint"},
    {"case_id": "x", "status": "FEASIBLE", "validator_status": "ACCEPTED", "variant": "joint", "primary": "29"},
])
def test_run_share_stops(row):
    with pytest.raises(a.AnalysisStop):
        a.run_share(row, {"input": {"snapshot": SNAP, "timetable": TT}})


def test_joint_recompute_match_passes():
    row = {"case_id": "x", "status": "FEASIBLE", "validator_status": "ACCEPTED", "variant": "joint", "primary": "30"}
    assert a.run_share(row, {"input": {"snapshot": SNAP, "timetable": TT}})[1] == "accepted"


def test_holm_and_wilcoxon():
    assert a.holm([0.01, 0.04]) == pytest.approx([0.02, 0.04])
    assert a.holm([None, 0.03]) == [None, 0.03]
    assert a.wilcoxon_greater([0.1] * 5 + [0.0] * 20) is None  # fewer than 6 non-zero pairs
    assert a.wilcoxon_greater([0.1, 0.2, 0.3, 0.15, 0.25, 0.05, 0.12]) < 0.05


def test_verdict_matrix():
    assert "не підтверджено" in a.verdict_text(False, True, "rotation")
    assert "менша за один урок" in a.verdict_text(True, False, "rotation")
    assert a.verdict_text(True, True, "rotation").startswith("спільна модель дає більше")
