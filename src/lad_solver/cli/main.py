"""``lad-solver`` command line: ``solve``, ``generate`` and ``worker``."""

import argparse
import importlib
import math
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from lad_solver import __version__
from lad_solver.app.solve import solve
from lad_solver.cli.generate import generate_problem
from lad_solver.cli.io import (
    problem_from_json,
    problem_to_json,
    read_json,
    result_to_json,
    write_json,
)
from lad_solver.domain.problem import Params, TimetableKind

EXIT_USAGE = 2


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    command: Callable[[argparse.Namespace], int] | None = getattr(args, "handler", None)
    if command is None:
        sys.stdout.write(f"lad-solver {__version__}\n")
        parser.print_help(sys.stdout)
        return 0
    return command(args)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lad-solver", description="Lad timetable solver")
    parser.add_argument("--version", action="version", version=f"lad-solver {__version__}")
    sub = parser.add_subparsers(title="commands")

    p_solve = sub.add_parser("solve", help="solve a problem JSON file into a result JSON file")
    p_solve.add_argument("--in", dest="input", type=Path, required=True)
    p_solve.add_argument("--out", type=Path, required=True)
    p_solve.add_argument("--time-limit", type=_positive_float, default=60.0)
    p_solve.add_argument("--workers", type=_positive_int, default=4)
    p_solve.add_argument("--seed", type=int, default=0)
    p_solve.add_argument("--deterministic-time", type=_positive_float, default=None)
    p_solve.add_argument("--interleave", action="store_true")
    p_solve.set_defaults(handler=_solve)

    p_gen = sub.add_parser("generate", help="generate a synthetic school problem")
    p_gen.add_argument("--classes", type=int, required=True)
    p_gen.add_argument("--shelter-ratio", type=_non_negative_float, required=True)
    p_gen.add_argument("--seed", type=int, default=0)
    p_gen.add_argument("--saturday", action="store_true")
    p_gen.add_argument("--kind", choices=[k.value for k in TimetableKind], default="main")
    p_gen.add_argument("--out", type=Path, required=True)
    p_gen.set_defaults(handler=_generate)

    p_worker = sub.add_parser("worker", help="run the Kafka worker")
    p_worker.set_defaults(handler=_worker)
    return parser


def _finite_float(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        msg = f"invalid number: {text!r}"
        raise argparse.ArgumentTypeError(msg) from None
    if not math.isfinite(value):
        msg = f"must be a finite number: {text!r}"
        raise argparse.ArgumentTypeError(msg)
    return value


def _positive_float(text: str) -> float:
    value = _finite_float(text)
    if value <= 0:
        msg = f"must be > 0: {text!r}"
        raise argparse.ArgumentTypeError(msg)
    return value


def _non_negative_float(text: str) -> float:
    value = _finite_float(text)
    if value < 0:
        msg = f"must be >= 0: {text!r}"
        raise argparse.ArgumentTypeError(msg)
    return value


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        msg = f"invalid integer: {text!r}"
        raise argparse.ArgumentTypeError(msg) from None
    if value < 1:
        msg = f"must be >= 1: {text!r}"
        raise argparse.ArgumentTypeError(msg)
    return value


def _fail(command: str, message: object) -> int:
    sys.stderr.write(f"lad-solver {command}: {message}\n")
    return EXIT_USAGE


def _check_out_dir(command: str, out: Path) -> int | None:
    if out.parent.is_dir():
        return None
    return _fail(command, f"output directory does not exist: {out.parent}")


def _solve(args: argparse.Namespace) -> int:
    if (code := _check_out_dir("solve", args.out)) is not None:
        return code
    # Only the input and output steps are caught: a failure inside solve() is a bug and propagates.
    try:
        problem = problem_from_json(read_json(args.input))
    except (OSError, ValueError, TypeError) as exc:
        return _fail("solve", exc)
    params = Params(
        time_limit_s=args.time_limit,
        num_workers=args.workers,
        random_seed=args.seed,
        max_deterministic_time=args.deterministic_time,
        interleave_search=args.interleave,
    )
    result = solve(problem, params)
    try:
        write_json(args.out, result_to_json(result))
    except OSError as exc:
        return _fail("solve", exc)
    stats = result.stats
    sys.stdout.write(
        f"{result.status.value} objective={stats.objective} bound={stats.best_bound} "
        f"wall={stats.wall_time_s:.1f}s relaxed={stats.quality_relaxed}\n"
    )
    return 0


def _generate(args: argparse.Namespace) -> int:
    if (code := _check_out_dir("generate", args.out)) is not None:
        return code
    try:
        problem = generate_problem(
            args.classes,
            args.shelter_ratio,
            seed=args.seed,
            saturday=args.saturday,
            kind=TimetableKind(args.kind),
        )
    except ValueError as exc:
        return _fail("generate", exc)
    try:
        write_json(args.out, problem_to_json(problem))
    except OSError as exc:
        return _fail("generate", exc)
    return 0


# Third-party packages of the ``worker`` extra; any other ImportError is a bug and propagates.
_WORKER_EXTRA = frozenset({"confluent_kafka", "pydantic", "pydantic_core"})


def _worker(_args: argparse.Namespace) -> int:
    try:
        module = importlib.import_module("lad_solver.adapters.kafka.main")
    except ImportError as exc:
        if (exc.name or "").split(".", 1)[0] not in _WORKER_EXTRA:
            raise
        sys.stderr.write(
            f"lad-solver worker: Kafka adapter is unavailable ({exc.name}); "
            "install the 'worker' extra\n"
        )
        return EXIT_USAGE
    run: Callable[[], int] = module.run
    return run()
