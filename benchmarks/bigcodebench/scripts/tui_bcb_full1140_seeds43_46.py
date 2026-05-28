#!/usr/bin/env python3
import argparse
import curses
import os
import sys
import time
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

os.environ.setdefault(
    "GORSA_SWEEP_ROOT_TEMPLATE",
    "/workspace/runs/bcb_full_instruct_llama3_8b_temp12_full1140_n10_tok768_seed{seed}/gorsa",
)

from monitor_seed_sweep import render_once  # noqa: E402


def draw(stdscr, seeds: list[int], interval: float) -> None:
    curses.curs_set(0)
    stdscr.nodelay(True)
    next_refresh = 0.0
    cached = ""
    status = ""

    while True:
        now = time.time()
        if now >= next_refresh:
            try:
                cached = render_once(seeds)
                status = ""
            except Exception as exc:
                status = f"render error: {exc}"
            next_refresh = now + interval

        h, w = stdscr.getmaxyx()
        stdscr.erase()
        title = f"BigCodeBench full1140 seed sweep TUI | q quit | r refresh | +/- interval ({interval:.0f}s)"
        stdscr.addnstr(0, 0, title, max(0, w - 1), curses.A_REVERSE)
        for row, line in enumerate(cached.splitlines(), start=2):
            if row >= h - 2:
                break
            stdscr.addnstr(row, 0, line, max(0, w - 1))
        if status and h > 1:
            stdscr.addnstr(h - 1, 0, status, max(0, w - 1), curses.A_BOLD)
        stdscr.refresh()

        ch = stdscr.getch()
        if ch in (ord("q"), ord("Q")):
            return
        if ch in (ord("r"), ord("R")):
            next_refresh = 0.0
        elif ch in (ord("+"), ord("=")):
            interval = min(120.0, interval + 5.0)
            next_refresh = 0.0
        elif ch in (ord("-"), ord("_")):
            interval = max(2.0, interval - 5.0)
            next_refresh = 0.0
        time.sleep(0.1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Curses TUI for the full1140 seed 43-46 sweep.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[43, 44, 45, 46])
    parser.add_argument("--interval", type=float, default=15.0)
    args = parser.parse_args()
    curses.wrapper(draw, args.seeds, args.interval)


if __name__ == "__main__":
    main()
