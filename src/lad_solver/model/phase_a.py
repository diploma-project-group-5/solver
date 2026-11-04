"""Phase A: lessons per row and day, day formats, capacity and the whole objective (ADR §1).

Variables: ``y[r,d]`` lessons of row r on day d; ``f[c,d]`` class c is in person on day d;
``z[t,d]`` teacher t is on site; ``b[c,t,d]`` the pair (c,t) has lessons on d; ``q[c,d]`` priority
lessons of c held in person on d. Lesson numbers are left to phase B, which is always feasible
because class-day and teacher-day loads are <= L (König's edge-colouring theorem).
"""

from dataclasses import dataclass, field

from ortools.sat.python import cp_model

from lad_solver.domain.problem import AlertLimits
from lad_solver.domain.result import SolveStatus
from lad_solver.model.cpsat import RunInfo, SolverConfig, run
from lad_solver.model.index import Index
from lad_solver.model.stop import StopHandle

type RowDay = tuple[int, int]  # (row position, day)
type ClassDay = tuple[int, int]  # (class position, day)


@dataclass(frozen=True, slots=True)
class DayPlan:
    """A phase A solution."""

    loads: dict[RowDay, int]  # only positive loads
    in_person: frozenset[ClassDay]


@dataclass(frozen=True, slots=True)
class Score:
    primary: int
    secondary: int
    objective: int


@dataclass(frozen=True, slots=True)
class PhaseAOutcome:
    info: RunInfo
    plan: DayPlan | None


@dataclass(slots=True)
class PhaseAModel:
    model: cp_model.CpModel = field(default_factory=cp_model.CpModel)
    y: dict[RowDay, cp_model.IntVar] = field(default_factory=dict[RowDay, cp_model.IntVar])
    f: dict[ClassDay, cp_model.IntVar] = field(default_factory=dict[ClassDay, cp_model.IntVar])
    z: dict[tuple[int, int], cp_model.IntVar] = field(
        default_factory=dict[tuple[int, int], cp_model.IntVar]
    )
    b: dict[tuple[int, int, int], cp_model.IntVar] = field(
        default_factory=dict[tuple[int, int, int], cp_model.IntVar]
    )


def score(index: Index, plan: DayPlan) -> Score:
    """Objective of a plan recomputed from its values (exact even when q is slack in the model)."""
    primary = 0
    for (r, d), n in plan.loads.items():
        c = index.row_class[r]
        if index.row_priority[r] and (c, d) in plan.in_person:
            primary += index.students[c] * n
    secondary = sum(1 for c, _ in plan.in_person if index.class_priority[c])
    return Score(primary, secondary, index.weight * primary + secondary)


def solve_phase_a(
    index: Index,
    caps: tuple[int, ...],
    config: SolverConfig,
    stop: StopHandle | None = None,
    hint: DayPlan | None = None,
) -> PhaseAOutcome:
    built = build_phase_a(index, caps, hint)
    solver, info = run(built.model, config, stop)
    if info.status not in {SolveStatus.OPTIMAL, SolveStatus.FEASIBLE}:
        return PhaseAOutcome(info, None)
    loads = {k: v for k, var in built.y.items() if (v := int(solver.value(var))) > 0}
    in_person = frozenset(k for k, var in built.f.items() if solver.boolean_value(var))
    return PhaseAOutcome(info, DayPlan(loads, in_person))


def build_phase_a(index: Index, caps: tuple[int, ...], hint: DayPlan | None = None) -> PhaseAModel:
    m = PhaseAModel()
    _add_loads(m, index, caps)
    _add_formats(m, index)
    _add_teacher_presence(m, index)
    _add_capacity(m, index)
    _add_outside_shelter(m, index, caps)
    _add_objective(m, index, caps)
    _add_hints(m, index, hint)
    return m


def _add_loads(m: PhaseAModel, index: Index, caps: tuple[int, ...]) -> None:
    for r, hours in enumerate(index.row_hours):
        for d in index.days:
            ub = index.upper_bound(r, d, caps)
            if ub > 0:
                m.y[r, d] = m.model.new_int_var(0, ub, f"y_{r}_{d}")
        m.model.add(sum(m.y[r, d] for d in index.days if (r, d) in m.y) == hours)
    limit = index.lessons_per_day
    for groups in (index.rows_by_class, index.rows_by_teacher):
        for rows in groups:
            for d in index.days:
                day_vars = [m.y[r, d] for r in rows if (r, d) in m.y]
                if day_vars:
                    m.model.add(sum(day_vars) <= limit)


def _add_formats(m: PhaseAModel, index: Index) -> None:
    for c, rows in enumerate(index.rows_by_class):
        for d in index.days:
            day_vars = [m.y[r, d] for r in rows if (r, d) in m.y]
            if day_vars:  # an empty day is never in person: f <= sum(y)
                f = m.f[c, d] = m.model.new_bool_var(f"f_{c}_{d}")
                m.model.add(f <= sum(day_vars))


def _add_teacher_presence(m: PhaseAModel, index: Index) -> None:
    pairs_by_teacher: dict[int, list[int]] = {}
    for c, t in index.pair_rows:
        pairs_by_teacher.setdefault(t, []).append(c)
    for t, classes in pairs_by_teacher.items():
        for d in index.days:
            together = [_pair_has_lessons(m, index, c, t, d) for c in classes]
            active = [(c, b) for c, b in zip(classes, together, strict=True) if b is not None]
            if not active:
                continue
            z = m.z[t, d] = m.model.new_bool_var(f"z_{t}_{d}")
            for c, b in active:  # one format per teacher and day: b => z == f (linearised)
                f = m.f[c, d]
                m.model.add(z >= f + b - 1)
                m.model.add(z <= f + 1 - b)
            m.model.add(z <= sum(b for _, b in active))


def _pair_has_lessons(
    m: PhaseAModel, index: Index, c: int, t: int, d: int
) -> cp_model.IntVar | None:
    day_vars = [m.y[r, d] for r in index.pair_rows[c, t] if (r, d) in m.y]
    if not day_vars:
        return None
    b = m.b[c, t, d] = m.model.new_bool_var(f"b_{c}_{t}_{d}")
    total = sum(day_vars)
    m.model.add(total >= b)  # b => at least one lesson
    m.model.add(total <= index.lessons_per_day * b)  # not b => no lessons
    return b


def _add_capacity(m: PhaseAModel, index: Index) -> None:
    capacity = index.problem.capacity
    staff = index.problem.staff
    for d in index.days:
        students = sum(index.students[c] * f for (c, day), f in m.f.items() if day == d)
        teachers = sum(z for (_, day), z in m.z.items() if day == d)
        if isinstance(capacity, AlertLimits):
            m.model.add(students <= capacity.seats).with_name(f"seats_{d}")
        m.model.add(students + staff + teachers <= capacity.passport).with_name(f"passport_{d}")


def _add_outside_shelter(m: PhaseAModel, index: Index, caps: tuple[int, ...]) -> None:
    """ALERT only: a no-shelter lesson goes to a remote day (y <= cap * (1 - f))."""
    for (r, d), y in m.y.items():
        if index.row_outside[r]:
            f = m.f[index.row_class[r], d]
            m.model.add(y <= index.upper_bound(r, d, caps) * (1 - f))


def _add_objective(m: PhaseAModel, index: Index, caps: tuple[int, ...]) -> None:
    terms: list[cp_model.IntVar] = []
    coefficients: list[int] = []
    for c, rows in enumerate(index.rows_by_class):
        for q in _priority_in_person(m, index, caps, c, rows):
            terms.append(q)
            coefficients.append(index.weight * index.students[c])
    for (c, _), f in m.f.items():
        if index.class_priority[c]:
            terms.append(f)
            coefficients.append(1)
    m.model.maximize(sum(k * v for v, k in zip(terms, coefficients, strict=True)))


def _priority_in_person(
    m: PhaseAModel, index: Index, caps: tuple[int, ...], c: int, rows: tuple[int, ...]
) -> list[cp_model.IntVar]:
    """q[c,d] <= priority lessons of c on d; q[c,d] <= M_c * f[c,d]."""
    prio_rows = [r for r in rows if index.row_priority[r]]
    m_c = min(index.lessons_per_day, sum(min(caps[r], index.row_hours[r]) for r in prio_rows))
    if m_c == 0 or index.students[c] == 0:
        return []
    q_vars: list[cp_model.IntVar] = []
    for d in index.days:
        prio = [m.y[r, d] for r in prio_rows if (r, d) in m.y]
        if not prio:
            continue
        q = m.model.new_int_var(0, m_c, f"q_{c}_{d}")
        m.model.add(q <= sum(prio))
        m.model.add(q <= m_c * m.f[c, d])
        q_vars.append(q)
    m.model.add(sum(q_vars) <= sum(index.row_hours[r] for r in prio_rows))  # redundant cut
    return q_vars


def _add_hints(m: PhaseAModel, index: Index, hint: DayPlan | None) -> None:
    if hint is None:  # "everything remote" is always feasible for capacity
        for var in (*m.f.values(), *m.z.values()):
            m.model.add_hint(var, 0)
        return
    for key, y in m.y.items():
        m.model.add_hint(y, hint.loads.get(key, 0))
    for key, f in m.f.items():
        m.model.add_hint(f, int(key in hint.in_person))
    for (c, t, d), b in m.b.items():
        busy = any(hint.loads.get((r, d), 0) > 0 for r in index.pair_rows[c, t])
        m.model.add_hint(b, int(busy))
    for (t, d), z in m.z.items():
        on_site = any(
            hint.loads.get((r, d), 0) > 0 and (index.row_class[r], d) in hint.in_person
            for r in index.rows_by_teacher[t]
        )
        m.model.add_hint(z, int(on_site))
