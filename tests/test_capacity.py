"""``model/capacity``: one day-capacity rule for the build and the substitution model.

Moving the rule out of phase A (ADR 2026-11-15-solver-substitution-engine) must not change a single
byte of the build: the CP-SAT model of the platform seed school is compared with its text form
recorded before the move. CP-SAT is a function of the model and the parameters, so the same model
gives the same timetable on the same solver build.
"""

import hashlib

import pytest
from ortools.sat.python import cp_model

from lad_solver.cli.generate import generate_problem
from lad_solver.cli.io import canonical_sha256, problem_to_json
from lad_solver.domain.problem import AlertLimits, Passport, TimetableKind
from lad_solver.model.capacity import (
    DayRoom,
    add_day_capacity,
    baseline_room,
    day_people,
    day_room,
)
from lad_solver.model.cpsat import SolverConfig, run
from lad_solver.model.index import build_index
from lad_solver.model.phase_a import build_phase_a

# platform/seed/solver-problem.json is `lad-solver generate --classes 30 --shelter-ratio 0.55
# --seed 0` (its input_sha256 is in platform/seed/timetable-main.json); alert = the same school.
SEED_INPUT_SHA256 = {
    TimetableKind.MAIN: "b3b4d07e42b9c6efc95ff13ce820f4ba3bb6e361750da195628bc94713349750",
    TimetableKind.ALERT: "a806b74baca2b1b0418564f3b4712cbebd6145646f7307388770428f5304a89f",
}
# sha256 of str(model.proto) of phase A, recorded on solver 0.3.0 before the move (ortools 9.15).
PHASE_A_MODEL_SHA256 = {
    (TimetableKind.MAIN, False): "09d8f848c37269a312d6de49461e518ec09a8dba9fb7a2ac4dd4a94231bc311d",
    (TimetableKind.MAIN, True): "85d5036ea79860e826f0d62abc99a1245f67988d82599c5a04dbfe4bede8ae88",
    (
        TimetableKind.ALERT,
        False,
    ): "8bc014dfeed0f98449a3312c0ecdd6f9345e1dab9d81113dc6770bbc54490bfe",
    (TimetableKind.ALERT, True): "879cee2ca7ba78ab33ab55e5eccc235e032884a2f1bebc7be73d5de66cdabca4",
}


@pytest.mark.parametrize(("kind", "relaxed"), sorted(PHASE_A_MODEL_SHA256))
def test_phase_a_model_of_the_seed_school_is_byte_identical(
    kind: TimetableKind, *, relaxed: bool
) -> None:
    problem = generate_problem(30, 0.55, seed=0, kind=kind)
    assert canonical_sha256(problem_to_json(problem)) == SEED_INPUT_SHA256[kind]
    index = build_index(problem)

    built = build_phase_a(index, index.row_caps(2, relaxed=relaxed))

    text = str(built.model.proto)
    assert hashlib.sha256(text.encode()).hexdigest() == PHASE_A_MODEL_SHA256[kind, relaxed]
    assert ("seats_1" in text) == (kind is TimetableKind.ALERT)


def test_build_room_moves_staff_to_the_right_and_clamps_huge_limits() -> None:
    assert day_room(Passport(60), 4, max_pupils=40, max_teachers=5) == DayRoom(45, None)
    assert day_room(AlertLimits(30, 60), 4, max_pupils=40, max_teachers=5) == DayRoom(45, 30)
    huge = 2**63 - 1
    assert day_room(AlertLimits(huge, huge), 4, max_pupils=40, max_teachers=5) == DayRoom(45, 40)


def test_baseline_room_lets_a_stale_day_stay_but_not_grow() -> None:
    # passport 66: a day at 66 has no room for one more person, a day at 70 keeps its 70.
    assert baseline_room(Passport(66), 2, 66, max_pupils=60, max_teachers=5).people == 64
    assert baseline_room(Passport(66), 2, 70, max_pupils=60, max_teachers=10).people == 68
    # Formats are fixed in a substitution: the seats condition is never constrained.
    assert baseline_room(AlertLimits(60, 66), 2, 66, max_pupils=60, max_teachers=5).pupils is None


def test_day_people_is_pupils_staff_and_teachers() -> None:
    assert day_people(60, 2, 4) == 66


@pytest.mark.parametrize(
    ("teachers_on_site", "feasible"), [(4, True), (5, False)], ids=["at-limit", "over-by-one"]
)
def test_the_constraint_holds_exactly_at_the_limit(
    teachers_on_site: int, *, feasible: bool
) -> None:
    """60 pupils + 2 staff + teachers <= passport 66 (testing.md: at the limit passes, +1 fails)."""
    model = cp_model.CpModel()
    on_site = [model.new_bool_var(f"z{i}") for i in range(teachers_on_site)]
    for z in on_site:
        model.add(z == 1)
    room = day_room(AlertLimits(60, 66), 2, max_pupils=60, max_teachers=teachers_on_site)

    add_day_capacity(model, 1, 60, sum(on_site), room)

    _, info = run(model, SolverConfig(time_limit_s=5), has_objective=False)
    assert (info.status.value in {"OPTIMAL", "FEASIBLE"}) == feasible
