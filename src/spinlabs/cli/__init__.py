"""Command line interface for spinlabs.

Each module in this package defines one CLI domain, e.g. plotting.
"""

import argparse

from spinlabs.logging_utils import configure_logging


def add_verbosity_args(parser: argparse.ArgumentParser) -> None:
    """Add the shared ``-v``/``-q`` console logging flags to a subcommand."""
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="also show debug output: how each step is done (per-mode fits, settings, ...)",
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="Show only warnings and errors")


def build_parser() -> argparse.ArgumentParser:
    from spinlabs.cli.listing import build_list_parser
    from spinlabs.cli.package import build_package_parser
    from spinlabs.cli.plotting import build_plot_parser
    from spinlabs.cli.run import build_run_parser
    from spinlabs.cli.tables import build_table_parser

    parser = argparse.ArgumentParser(prog="spinlabs", description="SpinLabs command line tools")
    subparsers = parser.add_subparsers(dest="domain", required=True)

    plot_parser = subparsers.add_parser("plot", help="Plotting commands")
    build_plot_parser(plot_parser.add_subparsers(dest="command", required=True))

    build_run_parser(subparsers.add_parser("run", help="Run the end-to-end analysis pipeline"))

    table_parser = subparsers.add_parser("table", help="LaTeX table commands")
    build_table_parser(table_parser.add_subparsers(dest="command", required=True))

    build_list_parser(
        subparsers.add_parser("list", help="When the latest packaged data and runs were made")
    )

    build_package_parser(subparsers.add_parser("package", help="Package (or with --scan, only parse) raw ZFS simulation data into a single .npz"))

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    # Configure console logging once at the entry point so every subcommand
    # (including ones without their own flags, e.g. `plot`) gets sane levels.
    verbosity = getattr(args, "verbose", 0)
    quiet = getattr(args, "quiet", False)
    configure_logging(verbosity=verbosity, quiet=quiet)
    args.func(args)
