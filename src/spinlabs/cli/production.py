"""The coupling and rate flags shared by ``spinlabs run`` and its stages (``coupling``, ``sigma-scan``).

Their defaults are the production settings of the paper figures (scripts/paper/config.py and the
PRODUCTION_* constants of spinlabs.pipeline); each has a flag that turns it off. Defining them once
here keeps the stages computing what ``spinlabs run`` computes.
"""

import argparse

from spinlabs.pipeline import ALL_PROCESSES, PRODUCTION_DIRECT, PRODUCTION_PROCESSES
from spinlabs.relaxation_dynamics import INIT_STATES, init_populations, init_states, normalize_init_state
from spinlabs.transition_rate import TransitionRate


def add_coupling_args(opt: argparse._ArgumentGroup) -> None:
    """ZFSManager options: inconsistent pairs excluded and the off-diagonal fill on by default."""
    opt.add_argument(
        "--exclude-inconsistent-pairs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop second-order entries whose ZFS change departs from the one expected from the first-order runs "
        "by a multiple of it (strongly nonlinear modes, or over-displaced or stale runs); they are listed either "
        "way (default: dropped; --no-exclude-inconsistent-pairs keeps them)",
    )
    # older spelling of --no-exclude-inconsistent-pairs
    opt.add_argument(
        "--keep-inconsistent-pairs", dest="exclude_inconsistent_pairs", action="store_false", help=argparse.SUPPRESS
    )
    opt.add_argument(
        "--degenerate-pairs-from-symmetry",
        action="store_true",
        help="Take the second-order coupling of each degenerate Ex-Ey pair from its Ex-Ex derivative (C3v "
        "symmetry) instead of the computed Ex-Ey entry, so those runs are not needed",
    )
    opt.add_argument(
        "--offdiag-from-diagonal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Approximate every second-order pair (i, j), i != j, that was not run from the diagonal couplings: "
        "sqrt(V_ii V_jj) per coupling type, and sqrt(V00_aa V_ee) for A x E pairs; a heuristic, only for the "
        "types the pair's symmetry allows (computed pairs keep their values) (default: on; "
        "--no-offdiag-from-diagonal turns it off)",
    )


def coupling_options(args: argparse.Namespace) -> dict:
    """ZFSManager keyword arguments from the flags of :func:`add_coupling_args`."""
    return dict(
        exclude_inconsistent_pairs=args.exclude_inconsistent_pairs,
        degenerate_pairs_from_symmetry=args.degenerate_pairs_from_symmetry,
        offdiag_from_diagonal=args.offdiag_from_diagonal,
    )


def add_rate_args(opt: argparse._ArgumentGroup, raman_note: str = "") -> None:
    """Rate options: Debye direct rate, zero-gap pairs left out, T1 without first-order Raman by default.

    ``raman_note`` is appended to the --first-order-raman help (what the command saves either way).
    """
    opt.add_argument(
        "--direct",
        choices=TransitionRate.DIRECT_MODELS,
        default=PRODUCTION_DIRECT,
        help="One-phonon (direct) rate: 'debye' J(E) = C E^3 fitted to the supercell's folded acoustic modes, "
        "'gaussian' the smeared J at the splitting (only Gaussian tails of optical modes), 'none' "
        f"(default: {PRODUCTION_DIRECT})",
    )
    opt.add_argument(
        "--keep-zero-gap-pairs",
        action="store_true",
        help="Keep the mode pairs at zero frequency gap by symmetry ((i, i) and degenerate partners) in the "
        "two-phonon rates; by default they are left out (their weight ~ 1/sigma has no continuum counterpart)",
    )
    opt.add_argument(
        "--first-order-raman",
        action="store_true",
        help="Also solve T1 with the first-order Raman rate (first-order couplings at second order); by default "
        "T1 comes from the direct and two-phonon rates only" + (f". {raman_note}" if raman_note else ""),
    )


def rate_options(args: argparse.Namespace) -> dict:
    """``direct``, ``nogap`` and ``processes`` (run_full_pipeline/compute_rates) from :func:`add_rate_args`."""
    return dict(
        direct=args.direct,
        nogap=not args.keep_zero_gap_pairs,
        processes=ALL_PROCESSES if args.first_order_raman else PRODUCTION_PROCESSES,
    )


def add_init_state_arg(opt: argparse._ArgumentGroup, help_prefix: str) -> None:
    """``--init-state``: names of any spin; checked against the couplings' spin once they are computed."""
    opt.add_argument(
        "--init-state",
        nargs="+",
        type=normalize_init_state,
        default=list(INIT_STATES),
        help=f"{help_prefix} (default: every state of the spin: ms_1 ms_0 ms_-1 for S = 1, also given as 1 0 -1; "
        "ms_pm1_2 ms_pm3_2 for S = 3/2)",
    )


def resolve_init_states(states: list[str], spin: float) -> list[str]:
    """The initial states to solve for ``spin``, checked before any rate is computed, as `spinlabs run` does.

    The S = 1 default (INIT_STATES) becomes every state of ``spin``; any other name must be a state of ``spin``
    (relaxation_dynamics.init_populations), for S = 1 too, else ValueError listing the valid states.
    """
    states = list(states)
    if float(spin) != 1.0 and states == list(INIT_STATES):
        states = list(init_states(spin))
    valid = init_populations(spin)
    unknown = [s for s in states if s not in valid]
    if unknown:
        raise ValueError(f"init_state {', '.join(unknown)} is not a state of spin {spin:g}: {', '.join(valid)}")
    return states
