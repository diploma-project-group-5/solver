"""Integer-indexed view of a validated ``Problem`` shared by both phases."""

from collections import defaultdict
from dataclasses import dataclass

from lad_solver.domain.prechecks import available_days, row_day_cap
from lad_solver.domain.problem import Problem, TimetableKind


@dataclass(frozen=True, slots=True)
class Index:
    problem: Problem
    days: tuple[int, ...]
    lessons_per_day: int
    students: tuple[int, ...]  # per class position
    class_priority: tuple[bool, ...]
    row_class: tuple[int, ...]  # per row position
    row_teacher: tuple[int, ...]
    row_hours: tuple[int, ...]
    row_priority: tuple[bool, ...]  # subject of a priority educational area
    row_outside: tuple[bool, ...]  # "not held in the shelter" and the timetable is ALERT
    rows_by_class: tuple[tuple[int, ...], ...]
    rows_by_teacher: tuple[tuple[int, ...], ...]
    pair_rows: dict[tuple[int, int], tuple[int, ...]]  # (class, teacher) -> rows
    teacher_days: tuple[frozenset[int], ...]  # available school days per teacher

    @property
    def priority_classes(self) -> int:
        return sum(self.class_priority)

    @property
    def weight(self) -> int:
        """W = max(secondary) + 1 = |priority classes| * |D| + 1 (spec §04)."""
        return self.priority_classes * len(self.days) + 1

    def row_caps(self, cap_min: int, *, relaxed: bool) -> tuple[int, ...]:
        """Per-row daily cap; ``relaxed`` drops it to ``h_r`` (quality ladder)."""
        if relaxed:
            return self.row_hours
        return tuple(
            row_day_cap(h, len(self.teacher_days[t]), cap_min)
            for h, t in zip(self.row_hours, self.row_teacher, strict=True)
        )

    def upper_bound(self, row: int, day: int, caps: tuple[int, ...]) -> int:
        if day not in self.teacher_days[self.row_teacher[row]]:
            return 0
        return min(caps[row], self.lessons_per_day, self.row_hours[row])


def build_index(problem: Problem) -> Index:
    class_pos = {c.id: i for i, c in enumerate(problem.classes)}
    teacher_pos = {t.id: i for i, t in enumerate(problem.teachers)}
    subjects = {s.id: s for s in problem.subjects}
    alert = problem.kind is TimetableKind.ALERT
    rows = problem.rows
    by_class: list[list[int]] = [[] for _ in problem.classes]
    by_teacher: list[list[int]] = [[] for _ in problem.teachers]
    pairs: defaultdict[tuple[int, int], list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        c, t = class_pos[r.class_id], teacher_pos[r.teacher_id]
        by_class[c].append(i)
        by_teacher[t].append(i)
        pairs[c, t].append(i)
    return Index(
        problem=problem,
        days=problem.school_days,
        lessons_per_day=problem.lessons_per_day,
        students=tuple(c.students for c in problem.classes),
        class_priority=tuple(c.priority for c in problem.classes),
        row_class=tuple(class_pos[r.class_id] for r in rows),
        row_teacher=tuple(teacher_pos[r.teacher_id] for r in rows),
        row_hours=tuple(r.hours for r in rows),
        row_priority=tuple(subjects[r.subject_id].priority_area for r in rows),
        row_outside=tuple(alert and subjects[r.subject_id].no_shelter for r in rows),
        rows_by_class=tuple(tuple(x) for x in by_class),
        rows_by_teacher=tuple(tuple(x) for x in by_teacher),
        pair_rows={k: tuple(v) for k, v in pairs.items()},
        teacher_days=tuple(frozenset(available_days(problem, t)) for t in problem.teachers),
    )
