"""Lightweight CUDA-event profiler for expert offload pipeline."""
from __future__ import annotations

import contextlib
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Generator

import torch


@dataclass
class PhaseStats:
    total_ms: float = 0.0
    count: int = 0

    @property
    def mean_ms(self) -> float:
        return self.total_ms / self.count if self.count else 0.0


class OffloadProfiler:
    """Records CPU wall-time and CUDA-event durations for each phase.

    Usage:
        profiler = OffloadProfiler(device)
        with profiler.phase("h2d_transfer"):
            buf.copy_(data, non_blocking=True)
        profiler.report()

    Two timing modes:
        mode="cpu"  — time.perf_counter() (low overhead; measures CPU dispatch
                       time plus any blocking; safe to use in production runs)
        mode="cuda" — torch.cuda.Event pairs with explicit synchronise (accurate
                       GPU kernel time; high overhead; profiling runs only)
    """

    def __init__(
        self,
        device: torch.device,
        enabled: bool = True,
        mode: str = "cpu",
    ):
        self.device = device
        self.enabled = enabled
        self.mode = mode
        self._stats: dict[str, PhaseStats] = defaultdict(PhaseStats)
        self._token_times: list[float] = []
        self._token_start: float = 0.0
        # hit/miss counters accumulated from the pipeline
        self.total_hits: int = 0
        self.total_misses: int = 0

    # ------------------------------------------------------------------
    # Token-level timing
    # ------------------------------------------------------------------

    def begin_token(self) -> None:
        if not self.enabled:
            return
        self._token_start = time.perf_counter()

    def end_token(self) -> None:
        if not self.enabled:
            return
        self._token_times.append(time.perf_counter() - self._token_start)

    # ------------------------------------------------------------------
    # Phase recording
    # ------------------------------------------------------------------

    def record_ms(self, phase: str, ms: float) -> None:
        """Accumulate a pre-measured duration (milliseconds) for *phase*."""
        if not self.enabled:
            return
        s = self._stats[phase]
        s.total_ms += ms
        s.count += 1

    @contextlib.contextmanager
    def phase(self, name: str) -> Generator[None, None, None]:
        """Context manager that times the body and records the duration."""
        if not self.enabled:
            yield
            return

        if self.mode == "cuda":
            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt = torch.cuda.Event(enable_timing=True)
            start_evt.record()
            yield
            end_evt.record()
            torch.cuda.synchronize()
            self.record_ms(name, start_evt.elapsed_time(end_evt))
        else:
            t0 = time.perf_counter()
            yield
            self.record_ms(name, (time.perf_counter() - t0) * 1_000.0)

    # ------------------------------------------------------------------
    # Hit/miss accounting
    # ------------------------------------------------------------------

    def record_hits(self, hits: int, misses: int) -> None:
        if not self.enabled:
            return
        self.total_hits += hits
        self.total_misses += misses

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def report(self) -> str:
        if not self._token_times:
            return "(no profiling data)"

        import statistics

        n_tok = len(self._token_times)
        mean_tok = statistics.mean(self._token_times) * 1_000.0
        std_tok = (statistics.stdev(self._token_times) * 1_000.0) if n_tok > 1 else 0.0
        mean_tps = 1_000.0 / mean_tok if mean_tok > 0 else 0.0

        # token_total is derived from _token_times (measured around full model forward)
        token_total_ms = mean_tok

        lines: list[str] = []
        lines.append(f"\n=== Offload Profiler Report ({n_tok} tokens) ===")
        lines.append("")
        lines.append("Per-token summary:")
        lines.append(f"  Mean tok latency:  {mean_tok:>10.1f} ms")
        lines.append(f"  Stddev:            {std_tok:>10.1f} ms")
        lines.append("")

        # Build synthetic token_total PhaseStats from _token_times.
        token_total_stat = PhaseStats(
            total_ms=sum(t * 1_000.0 for t in self._token_times),
            count=n_tok,
        )

        # Sort phases: token_total first, then descending by total_ms
        other_phases = sorted(
            self._stats.keys(),
            key=lambda k: self._stats[k].total_ms,
            reverse=True,
        )
        all_phase_names = ["token_total"] + other_phases
        all_phases: dict[str, PhaseStats] = {"token_total": token_total_stat}
        all_phases.update(self._stats)

        col_w = max((len(p) for p in all_phase_names), default=8) + 2
        lines.append("Phase breakdown (mean per call, total calls):")
        header = (
            f"  {'Phase':<{col_w}}  {'Mean ms':>8}  {'Calls':>6}  {'Total ms':>10}  {'% of tok':>9}"
        )
        sep = "  " + "\u2500" * (col_w + 42)
        lines.append(header)
        lines.append(sep)

        for p in all_phase_names:
            s = all_phases[p]
            pct = (s.mean_ms / token_total_ms * 100) if token_total_ms > 0 else 0.0
            lines.append(
                f"  {p:<{col_w}}  {s.mean_ms:>8.1f}  {s.count:>6}  {s.total_ms:>10.1f}  {pct:>8.1f}%"
            )

        lines.append("")
        total_ops = self.total_hits + self.total_misses
        if total_ops > 0:
            hit_pct = self.total_hits / total_ops * 100
            miss_pct = self.total_misses / total_ops * 100
            lines.append("Expert hit/miss:")
            lines.append(f"  Hits:    {self.total_hits:>6} / {total_ops}  ({hit_pct:5.1f}%)")
            lines.append(f"  Misses:  {self.total_misses:>6} / {total_ops}  ({miss_pct:5.1f}%)")
            lines.append("")

        lines.append("Throughput:")
        lines.append(f"  Mean: {mean_tps:.2f} tok/s  (from profiled tokens, includes overhead)")

        return "\n".join(lines)
