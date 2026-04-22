"""PA2 grader — reads bench CSV, scores correctness + performance, prints."""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CorrectnessCase:
    name: str
    points: int


@dataclass(frozen=True)
class PerfCase:
    name: str
    baseline_cycles: int
    tier_speedups: tuple[float, ...]
    tier_points: tuple[int, ...]


CORRECTNESS: tuple[CorrectnessCase, ...] = (
    CorrectnessCase("fft_4", 6),
    CorrectnessCase("fft_8", 6),
    CorrectnessCase("fft_64", 6),
    CorrectnessCase("fft_512", 6),
    CorrectnessCase("fft_4096", 6),
    CorrectnessCase("power_zeros", 3),
    CorrectnessCase("power_1frame", 3),
    CorrectnessCase("power_small", 3),
    CorrectnessCase("power_medium", 3),
    CorrectnessCase("power_large", 3),
    CorrectnessCase("mel_tiny", 3),
    CorrectnessCase("mel_small", 3),
    CorrectnessCase("mel_medium", 3),
    CorrectnessCase("mel_dense", 3),
    CorrectnessCase("mel_large", 3),
)

PERF: tuple[PerfCase, ...] = (
    PerfCase("fft_4096", 1_590_000, (1.0, 1.1, 1.25, 1.45), (5, 10, 15, 20)),
    PerfCase("power_large", 113_000, (1.0, 2.5, 5.0, 10.0), (5, 10, 15, 20)),
    PerfCase("mel_large", 3_130_000, (1.0, 2.0, 4.0, 8.0), (5, 10, 15, 20)),
    PerfCase("melspec_trumpet", 24_600_000, (1.0, 1.2, 1.45, 1.75), (5, 10, 15, 20)),
)

CORRECTNESS_MAX = 60
PERF_MAX = 80


@dataclass(frozen=True)
class TestResult:
    name: str
    passed: bool
    cycles: int


@dataclass(frozen=True)
class CorrectnessRow:
    case: CorrectnessCase
    status: str  # "PASS" | "FAIL" | "MISS"
    points: int


@dataclass(frozen=True)
class PerfRow:
    case: PerfCase
    cycles: int | None  # None == MISS
    speedup: float  # 0.0 if missing/failed
    points: int
    failed: bool  # True iff result present but pass==0


@dataclass(frozen=True)
class ScoreReport:
    correctness_rows: tuple[CorrectnessRow, ...]
    perf_rows: tuple[PerfRow, ...]
    correctness_score: int
    perf_score: int

    @property
    def total(self) -> int:
        return self.correctness_score + self.perf_score


class Scorer:
    @staticmethod
    def parse_csv(path: Path) -> dict[str, TestResult]:
        out: dict[str, TestResult] = {}
        with path.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row["test_name"]
                out[name] = TestResult(
                    name=name,
                    passed=bool(int(row["pass"])),
                    cycles=int(row["cycles"]),
                )
        return out

    @staticmethod
    def _score_perf_tier(
        cycles: int,
        baseline: int,
        speedups: tuple[float, ...],
        points: tuple[int, ...],
    ) -> tuple[int, float]:
        if cycles == 0 or baseline == 0:
            return 0, 0.0
        speedup = baseline / cycles
        score = 0
        for s, p in zip(speedups, points):
            if speedup >= s:
                score = p
        return score, speedup

    @classmethod
    def score(cls, results: dict[str, TestResult]) -> ScoreReport:
        c_rows: list[CorrectnessRow] = []
        c_total = 0
        for case in CORRECTNESS:
            r = results.get(case.name)
            if r is None:
                c_rows.append(CorrectnessRow(case, "MISS", 0))
            elif r.passed:
                c_rows.append(CorrectnessRow(case, "PASS", case.points))
                c_total += case.points
            else:
                c_rows.append(CorrectnessRow(case, "FAIL", 0))

        p_rows: list[PerfRow] = []
        p_total = 0
        for case in PERF:
            r = results.get(case.name)
            if r is None:
                p_rows.append(PerfRow(case, None, 0.0, 0, False))
                continue
            if not r.passed or r.cycles == 0:
                p_rows.append(PerfRow(case, r.cycles, 0.0, 0, failed=True))
                continue
            pts, speedup = cls._score_perf_tier(
                r.cycles, case.baseline_cycles, case.tier_speedups, case.tier_points
            )
            p_rows.append(PerfRow(case, r.cycles, speedup, pts, False))
            p_total += pts

        return ScoreReport(tuple(c_rows), tuple(p_rows), c_total, p_total)


class Display:
    def __init__(self, console=None):
        from rich.console import Console

        self.console = console or Console()

    def render(self, report: ScoreReport) -> None:
        from rich.panel import Panel
        from rich.table import Table

        ct = Table(title=f"Correctness ({CORRECTNESS_MAX} pts)")
        ct.add_column("Test")
        ct.add_column("Result")
        ct.add_column("Points", justify="right")
        for row in report.correctness_rows:
            colour = {"PASS": "green", "FAIL": "red", "MISS": "yellow"}[row.status]
            ct.add_row(
                row.case.name,
                f"[{colour}]{row.status}[/]",
                f"{row.points}/{row.case.points}",
            )
        ct.add_section()
        ct.add_row(
            "[bold]Subtotal[/]",
            "",
            f"[bold]{report.correctness_score}/{CORRECTNESS_MAX}[/]",
        )
        self.console.print(ct)

        pt = Table(title=f"Performance ({PERF_MAX} pts)")
        pt.add_column("Test")
        pt.add_column("Cycles", justify="right")
        pt.add_column("Speedup", justify="right")
        pt.add_column("Points", justify="right")
        for row in report.perf_rows:
            cycles_str = "N/A" if row.cycles is None else f"{row.cycles:,}"
            speedup_str = "N/A" if row.cycles is None else f"{row.speedup:.2f}x"
            note = "  [red](FAIL — no perf credit)[/]" if row.failed else ""
            pt.add_row(
                row.case.name,
                cycles_str,
                speedup_str,
                f"{row.points}/{row.case.tier_points[-1]}{note}",
            )
        pt.add_section()
        pt.add_row(
            "[bold]Subtotal[/]", "", "", f"[bold]{report.perf_score}/{PERF_MAX}[/]"
        )
        self.console.print(pt)

        self.console.print(
            Panel(
                f"Correctness:  {report.correctness_score:3d} / {CORRECTNESS_MAX}\n"
                f"Performance:  {report.perf_score:3d} / {PERF_MAX}\n"
                f"[bold]Total:        {report.total:3d} / {CORRECTNESS_MAX + PERF_MAX}[/]",
                title="Final Score",
                expand=False,
            )
        )


def main() -> int:
    ap = argparse.ArgumentParser(description="Score a PA2 bench CSV.")
    ap.add_argument("csv", type=Path, help="Path to results CSV from `make run`.")
    args = ap.parse_args()
    report = Scorer.score(Scorer.parse_csv(args.csv))
    Display().render(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
