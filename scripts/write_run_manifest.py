#!/usr/bin/env python
"""Record code and input hashes for a locked TRACE run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trace_tfe3.utils.provenance import build_run_record, write_run_record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repository", default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args()
    write_run_record(build_run_record(args.input, args.repository), args.output)


if __name__ == "__main__":
    main()
