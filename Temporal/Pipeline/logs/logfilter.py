#!/usr/bin/env python3
"""
logfilter.py

Line:
  2026-01-07 12:44:10 [INFO] pipeline.temporal_fill.global: [global] s2_b12 coverage: 100.00%

- Filter by level(s): INFO, WARNING, ERROR, DEBUG, CRITICAL
- Filter by logger name substring (e.g. pipeline.whittaker)
- Filter by message substring or regex
- Filter by time window (--since / --until)
- Output modes:
    pretty
    raw
    json

- Stats:
    counts by level
    top loggers
    coverage summary extraction
    rows processed extraction

- Tail support for large files

=== USAGE ===
    python logfilter.py path/to/logfile.log [options]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Iterator, Optional, Pattern, Tuple

class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    RED = "\033[31m"
    YELLOW = "\033[33m"
    GREEN = "\033[32m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    GRAY = "\033[90m"

    @staticmethod
    def wrap(s: str, *codes: str, enable: bool = True) -> str:
        if not enable or not codes:
            return s
        return "".join(codes) + s + C.RESET


LEVEL_ORDER = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
LEVEL_COLOR = {
    "DEBUG": (C.GRAY,),
    "INFO": (C.GREEN,),
    "WARNING": (C.YELLOW, C.BOLD),
    "ERROR": (C.RED, C.BOLD),
    "CRITICAL": (C.MAGENTA, C.BOLD),
}

LOG_RE = re.compile(
    r"""
    ^
    (?P<ts>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})
    \s+\[(?P<level>[A-Z]+)\]
    \s+(?P<logger>[^:]+):
    \s+(?P<msg>.*)
    $
    """,
    re.VERBOSE,
)


@dataclass(frozen=True)
class LogLine:
    ts: datetime
    level: str
    logger: str
    msg: str
    raw: str


def parse_time(s: str) -> datetime:
    """
    Accepts:
      - "YYYY-MM-DD HH:MM:SS"
      - "YYYY-MM-DDTHH:MM:SS"
      - "YYYY-MM-DD"
    """
    s = s.strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    raise ValueError(f"Invalid time format: {s!r}")


def parse_line(line: str) -> Optional[LogLine]:
    m = LOG_RE.match(line.rstrip("\n"))
    if not m:
        return None
    try:
        ts = datetime.strptime(m.group("ts"), "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None
    level = m.group("level")
    logger = m.group("logger").strip()
    msg = m.group("msg")
    return LogLine(ts=ts, level=level, logger=logger, msg=msg, raw=line.rstrip("\n"))

def iter_lines(path: str, tail: Optional[int] = None) -> Iterator[str]:
    if tail is None:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                yield line
        return

    # tail last N lines efficiently
    # for giga/largo files: seek from end and read blocks.
    n = int(tail)
    if n <= 0:
        return

    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        end = f.tell()
        block_size = 8192
        data = b""
        lines = []
        pos = end

        while pos > 0 and len(lines) <= n:
            read_size = min(block_size, pos)
            pos -= read_size
            f.seek(pos)
            chunk = f.read(read_size)
            data = chunk + data
            lines = data.splitlines()

        # keep last n lines
        last = lines[-n:] if len(lines) >= n else lines
        for b in last:
            yield b.decode("utf-8", errors="replace") + "\n"

def level_pass(level: str, allowed: Optional[set[str]], min_level: Optional[str]) -> bool:
    if allowed is not None and level not in allowed:
        return False
    if min_level is None:
        return True
    if level not in LEVEL_ORDER or min_level not in LEVEL_ORDER:
        return True
    return LEVEL_ORDER.index(level) >= LEVEL_ORDER.index(min_level)


def time_pass(ts: datetime, since: Optional[datetime], until: Optional[datetime]) -> bool:
    if since and ts < since:
        return False
    if until and ts > until:
        return False
    return True


def logger_pass(logger: str, contains: Optional[str], regex: Optional[Pattern[str]]) -> bool:
    if contains and contains not in logger:
        return False
    if regex and not regex.search(logger):
        return False
    return True


def msg_pass(msg: str, contains: Optional[str], regex: Optional[Pattern[str]]) -> bool:
    if contains and contains not in msg:
        return False
    if regex and not regex.search(msg):
        return False
    return True

COVERAGE_RE = re.compile(r"\b(?P<col>[A-Za-z0-9_]+)\s+coverage:\s+(?P<pct>\d+(?:\.\d+)?)%")
ROWS_RE = re.compile(r"\b(?:Rows:\s+|->\s+)(?P<rows>\d+)\s+rows\b", re.IGNORECASE)


@dataclass
class Stats:
    total_lines: int = 0
    parsed_lines: int = 0
    matched_lines: int = 0
    parse_fail_lines: int = 0

    by_level_all: Counter = None
    by_level_matched: Counter = None
    by_logger_matched: Counter = None

    coverage: dict = None  # col -> list[pct]
    rows_values: list = None

    first_ts: Optional[datetime] = None
    last_ts: Optional[datetime] = None

    def __post_init__(self):
        self.by_level_all = Counter()
        self.by_level_matched = Counter()
        self.by_logger_matched = Counter()
        self.coverage = defaultdict(list)
        self.rows_values = []

    def observe_parsed(self, ll: LogLine):
        self.parsed_lines += 1
        self.by_level_all[ll.level] += 1
        if self.first_ts is None or ll.ts < self.first_ts:
            self.first_ts = ll.ts
        if self.last_ts is None or ll.ts > self.last_ts:
            self.last_ts = ll.ts

        # extract structured nuggets
        m = COVERAGE_RE.search(ll.msg)
        if m:
            col = m.group("col")
            pct = float(m.group("pct"))
            self.coverage[col].append(pct)

        r = ROWS_RE.search(ll.msg)
        if r:
            try:
                self.rows_values.append(int(r.group("rows")))
            except Exception:
                pass

    def observe_matched(self, ll: LogLine):
        self.matched_lines += 1
        self.by_level_matched[ll.level] += 1
        self.by_logger_matched[ll.logger] += 1

def format_pretty(ll: LogLine, color: bool = True, highlight: Optional[Pattern[str]] = None) -> str:
    ts = C.wrap(ll.ts.strftime("%Y-%m-%d %H:%M:%S"), C.DIM, enable=color)
    lvl_codes = LEVEL_COLOR.get(ll.level, (C.CYAN,))
    lvl = C.wrap(f"[{ll.level}]", *lvl_codes, enable=color)
    logger = C.wrap(ll.logger, C.BLUE, enable=color)

    msg = ll.msg
    if highlight:
        # underline matches (best-effort without relying on ANSI underline everywhere)
        def repl(m):
            return C.wrap(m.group(0), C.BOLD, enable=color)
        msg = highlight.sub(repl, msg)

    return f"{ts} {lvl} {logger}: {msg}"


def print_header(title: str, color: bool) -> None:
    line = "=" * len(title)
    print(C.wrap(title, C.BOLD, enable=color))
    print(C.wrap(line, C.DIM, enable=color))


def print_stats(stats: Stats, top_n: int, color: bool) -> None:
    print_header("Log Stats", color)

    print(f"Total lines read:    {stats.total_lines:,}")
    print(f"Parsed lines:        {stats.parsed_lines:,}")
    print(f"Matched lines:       {stats.matched_lines:,}")
    if stats.parse_fail_lines:
        print(C.wrap(f"Parse failures:      {stats.parse_fail_lines:,}", C.YELLOW, enable=color))

    if stats.first_ts and stats.last_ts:
        print(f"Time range:          {stats.first_ts}  ->  {stats.last_ts}")

    # levels
    print("\n" + C.wrap("Levels (matched)", C.BOLD, enable=color))
    for lvl in LEVEL_ORDER:
        count = stats.by_level_matched.get(lvl, 0)
        if count:
            lvl_s = C.wrap(lvl, *LEVEL_COLOR.get(lvl, ()), enable=color)
            print(f"  {lvl_s:<10} {count:,}")

    # top loggers
    print("\n" + C.wrap(f"Top loggers (matched, top {top_n})", C.BOLD, enable=color))
    for logger, count in stats.by_logger_matched.most_common(top_n):
        print(f"  {C.wrap(logger, C.BLUE, enable=color)}  {count:,}")

    # coverage summary
    if stats.coverage:
        print("\n" + C.wrap("Coverage snapshots (avg over matched lines that reported coverage)", C.BOLD, enable=color))
        rows = []
        for col, pcts in stats.coverage.items():
            if not pcts:
                continue
            avg = sum(pcts) / len(pcts)
            mn = min(pcts)
            mx = max(pcts)
            rows.append((avg, col, mn, mx, len(pcts)))
        rows.sort(reverse=True)

        for avg, col, mn, mx, n in rows[:top_n]:
            # color by avg
            if avg >= 99.0:
                cc = C.GREEN
            elif avg >= 90.0:
                cc = C.YELLOW
            else:
                cc = C.RED
            print(f"  {col:<18} {C.wrap(f'{avg:6.2f}%', cc, C.BOLD if avg < 90 else '', enable=color)}  "
                  f"(min {mn:6.2f}%, max {mx:6.2f}%, n={n})")

    # rows extracted
    if stats.rows_values:
        vals = stats.rows_values
        print("\n" + C.wrap("Row count mentions (from matched lines)", C.BOLD, enable=color))
        print(f"  count: {len(vals)} | min: {min(vals):,} | max: {max(vals):,} | last: {vals[-1]:,}")

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="logfilter",
        description="Filter MDR pipeline logs by level/logger/message with color + stats."
    )
    p.add_argument("file", help="Path to log file")
    p.add_argument("-L", "--level", action="append",
                   help="Include only these levels (repeatable). Example: -L INFO -L WARNING")
    p.add_argument("--min-level", choices=LEVEL_ORDER,
                   help="Include levels >= this severity (e.g., WARNING includes WARNING/ERROR/CRITICAL)")
    p.add_argument("--logger", help="Include only loggers containing this substring (case-sensitive)")
    p.add_argument("--logger-regex", help="Include only loggers matching this regex")
    p.add_argument("--contains", help="Include only lines whose message contains this substring (case-sensitive)")
    p.add_argument("--regex", help="Include only lines whose message matches this regex")
    p.add_argument("--since", help="Only include lines at/after this time (YYYY-MM-DD[ HH:MM:SS])")
    p.add_argument("--until", help="Only include lines at/before this time (YYYY-MM-DD[ HH:MM:SS])")
    p.add_argument("--tail", type=int, help="Read only the last N lines (fast for huge logs)")
    p.add_argument("--no-color", action="store_true", help="Disable ANSI colors")
    p.add_argument("--raw", action="store_true", help="Print raw lines instead of pretty formatting")
    p.add_argument("--json", action="store_true", help="Output matched lines as JSON objects (one per line)")
    p.add_argument("--stats", action="store_true", help="Print stats summary (recommended)")
    p.add_argument("--top", type=int, default=10, help="Top N for stats sections (default: 10)")
    p.add_argument("--highlight", action="store_true",
                   help="Highlight regex/contains matches in output (best-effort)")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_argparser().parse_args(argv)

    path = args.file
    if not os.path.exists(path):
        print(f"File not found: {path}", file=sys.stderr)
        return 2

    use_color = (not args.no_color) and sys.stdout.isatty()
    allowed_levels = set(args.level) if args.level else None
    min_level = args.min_level

    logger_re = re.compile(args.logger_regex) if args.logger_regex else None
    msg_re = re.compile(args.regex) if args.regex else None

    since = parse_time(args.since) if args.since else None
    until = parse_time(args.until) if args.until else None

    # highlight pattern
    highlight_pat = None
    if args.highlight:
        if msg_re:
            highlight_pat = msg_re
        elif args.contains:
            highlight_pat = re.compile(re.escape(args.contains))

    stats = Stats()

    # iterate
    for line in iter_lines(path, tail=args.tail):
        stats.total_lines += 1

        ll = parse_line(line)
        if ll is None:
            stats.parse_fail_lines += 1
            continue

        stats.observe_parsed(ll)

        if not time_pass(ll.ts, since, until):
            continue
        if not level_pass(ll.level, allowed_levels, min_level):
            continue
        if not logger_pass(ll.logger, args.logger, logger_re):
            continue
        if not msg_pass(ll.msg, args.contains, msg_re):
            continue

        stats.observe_matched(ll)

        if args.json:
            obj = {
                "ts": ll.ts.strftime("%Y-%m-%d %H:%M:%S"),
                "level": ll.level,
                "logger": ll.logger,
                "msg": ll.msg,
            }
            print(json.dumps(obj, ensure_ascii=False))
        elif args.raw:
            print(ll.raw)
        else:
            print(format_pretty(ll, color=use_color, highlight=highlight_pat))

    if args.stats:
        print()
        print_stats(stats, top_n=args.top, color=use_color)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
