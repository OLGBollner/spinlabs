"""Read a phonon file (phonopy .yaml or .npz) and save it as a .npz archive.

Usage:
    python read_phonons.py <phonon_file> [-o OUT_FILE]

If -o/--out is omitted, writes next to the input as <stem>.npz.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from spinlabs.parsers import (
    parse_phonon_data,
    prepare_phonopy_yaml,
    save_phonon_npz,
)


def _existing_phonon_file(value: str) -> Path:
    path = Path(value)
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"phonon file does not exist: {path}")
    if path.suffix.lower() not in {".yaml", ".yml", ".npz"}:
        raise argparse.ArgumentTypeError(f"unsupported phonon file type {path.suffix!r} (expected .yaml/.yml or .npz)")
    return path


def _writable_npz(value: str) -> Path:
    path = Path(value)
    if path.suffix.lower() != ".npz":
        raise argparse.ArgumentTypeError(f"output must end in .npz, got {path}")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Parse a phonon file (phonopy .yaml or .npz) and save it as a .npz archive."
    )
    parser.add_argument("phonon_file", type=_existing_phonon_file, help="phonopy .yaml or .npz phonon file")
    parser.add_argument(
        "-o",
        "--out",
        type=_writable_npz,
        default=None,
        help="output .npz path (default: <phonon_file stem>.npz next to the input)",
    )
    args = parser.parse_args(argv)

    out = args.out or args.phonon_file.with_suffix(".npz")

    try:
        if args.phonon_file.suffix in (".yaml", ".yml"):
            spectrum, _ = prepare_phonopy_yaml(args.phonon_file)
        else:
            spectrum = parse_phonon_data(args.phonon_file)
    except Exception as exc:  # surface a clean error, not a traceback
        print(f"error: failed to parse {args.phonon_file}: {exc}", file=sys.stderr)
        return 1

    save_phonon_npz(spectrum, out)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
