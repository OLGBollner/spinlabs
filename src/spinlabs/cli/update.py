"""CLI command ``spinlabs update``: set INCAR tags consistently across a defect's runs (spinlabs.vasp.incar_update)."""

import argparse
from pathlib import Path

from spinlabs.cli import add_verbosity_args


def build_update_parser(parser: argparse.ArgumentParser) -> None:
    from spinlabs.vasp.incar_update import ALLOWED_TAGS

    parser.add_argument("path", type=Path, help="defect folder (or any folder, or one INCAR); every INCAR under it")
    parser.add_argument(
        "--set",
        nargs="+",
        required=True,
        metavar="TAG=VALUE",
        help="tags to set, e.g. EDIFF=1E-7 ISYM=0 EDIFFG=-1E-7; only tags that must agree across all runs: "
        + ", ".join(sorted(ALLOWED_TAGS)),
    )
    parser.add_argument("--dry-run", action="store_true", help="print what would change and stop")
    parser.add_argument("-y", "--yes", action="store_true", help="write without asking")
    parser.add_argument("--skip-done", action="store_true", help="leave out finished runs (folders with .done)")
    add_verbosity_args(parser)
    parser.set_defaults(func=_run)


def _run(args: argparse.Namespace) -> None:
    from spinlabs.vasp.incar_update import apply_updates, find_incars, parse_settings, plan_updates, summary

    if not args.path.exists():
        raise SystemExit(f"error: not found: {args.path}")
    try:
        settings = parse_settings(args.set)
    except ValueError as error:
        raise SystemExit(f"error: {error}")
    incars = find_incars(args.path)
    if not incars:
        raise SystemExit(f"error: no INCAR under {args.path}")
    changes = plan_updates(args.path, settings, skip_done=args.skip_done)
    print(f"{len(incars)} INCAR(s) under {args.path}; setting " + " ".join(f"{t}={v}" for t, v in settings.items()))
    if not changes:
        print("nothing to change: every INCAR already has these values")
        return
    print("\n".join(summary(changes, args.path)))
    if args.dry_run:
        print("dry run: nothing written")
        return
    if not args.yes:
        try:
            answer = input(f"Write {len({c.path for c in changes})} INCAR(s)? (y/N): ").strip().lower()
        except EOFError:
            answer = ""
        if answer not in ("y", "yes"):
            print("nothing written")
            return
    written = apply_updates(changes)
    print(f"wrote {len(written)} INCAR(s)")
