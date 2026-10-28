"""``spinlabs run coupling``: the ZFSManager stage alone, for debugging.

Loads the inputs exactly as ``spinlabs run`` does, computes the first- and
(when present) second-order spin-phonon couplings, and prints a summary.
Nothing is written to disk or to the registry.
"""

import argparse

import numpy as np

from spinlabs.cli import add_verbosity_args
from spinlabs.constants import CONSTANTS
from spinlabs.logging_utils import get_logger, section
from spinlabs.registry import parse_orders

LOGGER = get_logger(__name__)

_V_KEYS = ("V_0_0", "V_p_m", "V_0_pm")


def add_input_args(parser: argparse.ArgumentParser) -> argparse._ArgumentGroup:
    """Input and coupling options shared by the run stages that start from raw ZFS data."""
    inputs = parser.add_argument_group("input mode (exactly one)")
    inputs.add_argument("--sim-folder", nargs="+", metavar="DIR", help="Path to VASP simulation folder(s)")
    inputs.add_argument("--raw-zfs-file", metavar="FILE", help="Path to a raw ZFS .npz dataset file")
    inputs.add_argument("--raw-zfs-file-1d", metavar="FILE", help="Raw ZFS 1st-order .npz (combined 1d+2d mode)")
    inputs.add_argument("--raw-zfs-file-2d", metavar="FILE", help="Raw ZFS 2nd-order .npz (combined 1d+2d mode)")

    opt = parser.add_argument_group("options")
    opt.add_argument("-ph", "--phonon-file", metavar="FILE", help="phonopy.yaml or phonon_data.npz")
    opt.add_argument(
        "--method",
        choices=["all", "approx"],
        help="Calculation method; inferred from the input name when omitted (as for run)",
    )
    opt.add_argument(
        "--order",
        nargs="+",
        type=int,
        choices=[1, 2, 12],
        help="Order(s) to compute (default: every order in the data). Second order needs first-order data",
    )
    opt.add_argument("--pert-scale", type=float, help="Override perturbation scale (as for run)")
    opt.add_argument("--defect", help="Override defect name")
    opt.add_argument("--cell-size", type=int, help="Override supercell size")
    opt.add_argument(
        "--exclude-inconsistent-pairs",
        action="store_true",
        help="Drop second-order entries departing from the first-order expectation (kept and listed by default)",
    )
    opt.add_argument("--keep-inconsistent-pairs", action="store_true", help=argparse.SUPPRESS)
    opt.add_argument(
        "--degenerate-pairs-from-symmetry",
        action="store_true",
        help="Take the second-order coupling of each degenerate Ex-Ey pair from its Ex-Ex derivative (as for run)",
    )
    opt.add_argument(
        "--offdiag-from-diagonal",
        action="store_true",
        help="Approximate every second-order pair (i, j), i != j, that was not run as sqrt(V_ii V_jj) from the "
        "diagonal couplings (computed pairs keep their values)",
    )
    return opt


def build_coupling_parser(parser: argparse.ArgumentParser) -> None:
    """Configure ``parser`` as the ``run coupling`` subcommand."""
    opt = add_input_args(parser)
    opt.add_argument("--top", type=int, default=10, help="Number of strongest modes/pairs to list (default: 10)")
    opt.add_argument(
        "--mode",
        nargs="+",
        type=int,
        default=[],
        metavar="N",
        help="Also list these modes (1-based, as in the -v derivative details and POSCAR generator)",
    )
    add_verbosity_args(opt)

    parser.set_defaults(func=_run_coupling)


def load_manager(args: argparse.Namespace, command: str):
    """ZFSManager for the inputs in ``args``, loaded as ``spinlabs run`` loads them.

    Returns ``(manager, phonon_path, orders)``; ``orders`` is what to compute.
    """
    from spinlabs.cli.run import _METHOD_MAP, _infer_method
    from spinlabs.pipeline import load_raw_inputs
    from spinlabs.zfs_manager import ZFSManager

    given = [
        flag
        for flag, val in (
            ("--sim-folder", args.sim_folder),
            ("--raw-zfs-file", args.raw_zfs_file),
            ("--raw-zfs-file-1d/2d", args.raw_zfs_file_1d or args.raw_zfs_file_2d),
        )
        if val
    ]
    if len(given) != 1:
        raise SystemExit(
            f"{command}: exactly one of --sim-folder, --raw-zfs-file, --raw-zfs-file-1d/2d is required"
            + (f" (got {', '.join(given)})" if given else "")
        )
    if (args.raw_zfs_file_1d is None) != (args.raw_zfs_file_2d is None):
        raise SystemExit(f"{command}: --raw-zfs-file-1d and --raw-zfs-file-2d must be given together")

    # _infer_method reads run's namespace, where --method and --raw-zfs-file are lists.
    method_args = argparse.Namespace(
        method=[args.method] if args.method else None,
        sim_folder=args.sim_folder,
        raw_zfs_file=[args.raw_zfs_file] if args.raw_zfs_file else None,
        raw_zfs_file_1d=args.raw_zfs_file_1d,
        raw_zfs_file_2d=args.raw_zfs_file_2d,
        coupling_file=None,
    )
    calc_method, zfs_folder = _METHOD_MAP[_infer_method(method_args)]
    sim_folder = args.sim_folder[0] if args.sim_folder and len(args.sim_folder) == 1 else args.sim_folder
    orders = parse_orders(args.order) if args.order else [1, 2]
    # A single sim folder holds one order; like run, take it from --order or the folder name.
    if len(orders) == 1:
        order_hint = orders[0]
    else:
        order_hint = 2 if any(k in str(sim_folder).lower() for k in ("second_order", "2d", "2nd")) else 1

    section(LOGGER, "Load input data")
    raw_data, spectrum, ph_path = load_raw_inputs(
        sim_folder=sim_folder,
        raw_zfs_file=args.raw_zfs_file,
        raw_zfs_file_1d=args.raw_zfs_file_1d,
        raw_zfs_file_2d=args.raw_zfs_file_2d,
        phonon_file=args.phonon_file,
        calc_method=calc_method,
        zfs_folder=zfs_folder,
        order=order_hint,
        pert_scale=args.pert_scale,
        defect=args.defect,
        cell_size=args.cell_size,
    )
    manager = ZFSManager(
        raw_data=raw_data,
        spectrum=spectrum,
        exclude_inconsistent_pairs=args.exclude_inconsistent_pairs,
        degenerate_pairs_from_symmetry=args.degenerate_pairs_from_symmetry,
        offdiag_from_diagonal=args.offdiag_from_diagonal,
    )
    manager.calc_method = calc_method
    if not manager.zfs_tensors:
        raise SystemExit("No first-order tensors in the input; every coupling needs the first-order derivatives.")
    if 2 in orders and not manager.zfs_tensors_2d and args.order:
        raise SystemExit("--order 2 requested, but the input has no second-order tensors.")
    return manager, ph_path, orders


def _run_coupling(args: argparse.Namespace) -> None:
    manager, ph_path, orders = load_manager(args, "spinlabs run coupling")
    section(LOGGER, f"Spin-phonon coupling: {manager.defect}_{manager.cell_size} {manager.calc_method}")
    first = manager.calculate_first_order_derivatives()
    second = None
    if 2 in orders and manager.zfs_tensors_2d:
        second = manager.calculate_second_order_derivatives(first[0])

    print(_summary(manager, ph_path, first, second, args.top, [m - 1 for m in args.mode]))


def _summary(manager, ph_path, first, second, top: int, modes: list[int]) -> str:
    """Readable report of the computed couplings (MHz per unit q, frequencies in meV)."""
    mhz = CONSTANTS["MHz2J"]
    pert = manager.get_phonon_pert(manager.pert_scale * CONSTANTS["ang_amu2SI"])
    freqs = np.asarray(pert["freqs"], dtype=float) / CONSTANTS["meV2J"] if pert else np.full(manager.nmodes, np.nan)
    syms = list(pert["sym"]) if pert else ["?"] * manager.nmodes
    iprs = np.asarray(pert["ipr"]) if pert else np.full(manager.nmodes, np.nan)
    d_zz = np.diag(manager.zfs_relaxed)[2] / mhz if manager.zfs_relaxed is not None else float("nan")
    n_diag = sum(1 for i, j in manager.zfs_tensors_2d if i == j)

    lines = [
        f"{manager.defect}_{manager.cell_size}  {manager.calc_method}  pert {manager.pert_scale:g}",
        f"  phonons: {ph_path}  ({manager.nmodes} modes, {len(manager.treated_modes)} treated)",
        f"  ground-state D_zz (principal frame): {d_zz:.3f} MHz",
        f"  tensors: {len(manager.zfs_tensors)} first order, {len(manager.zfs_tensors_2d)} second order "
        f"({n_diag} diagonal, {len(manager.zfs_tensors_2d) - n_diag} off-diagonal)",
    ]

    _, *v1 = first
    v1 = [np.asarray(v) / mhz for v in v1]
    lines.append("\nFirst order (MHz/q):")
    lines += [f"  {k:6s} nonzero {np.sum(v > 0):5d}   sum V^2 {np.sum(v**2):.4g}" for k, v in zip(_V_KEYS, v1)]
    strength = sum(v**2 for v in v1)
    picked = list(dict.fromkeys([*np.argsort(-strength)[:top].tolist(), *modes]))
    lines.append(f"  {'mode':>5s} {'sym':>3s} {'meV':>8s} {'IPR':>7s} " + " ".join(f"{k:>11s}" for k in _V_KEYS))
    for i in picked:
        lines.append(
            f"  {i + 1:5d} {syms[i]:>3s} {freqs[i]:8.2f} {iprs[i]:7.4f} " + " ".join(f"{v[i]:11.4g}" for v in v1)
        )

    if second is not None:
        _, *v2 = second
        v2 = [np.asarray(v) / mhz for v in v2]
        lines.append("\nSecond order (MHz/q^2):")
        bad = manager.inconsistent_pairs
        action = "kept" if not manager.exclude_inconsistent_pairs else "excluded"
        lines.append(
            f"  inconsistent entries {action}: {len(bad)} of {len(manager.zfs_tensors_2d)}"
            + (" -- " + ", ".join(f"{i + 1}_{j + 1} (alpha={a:.2f})" for i, j, a, _ in bad) if bad else "")
        )
        for k, v in zip(_V_KEYS, v2):
            total = np.sum(v**2)
            off = total - np.sum(np.diag(v) ** 2)
            share = off / total if total else 0.0
            lines.append(
                f"  V2{k[1:]:5s} nonzero {np.count_nonzero(np.triu(v)):6d}   sum V^2 {total:.4g}"
                f"   off-diagonal share {share:.1%}"
            )
        strength2 = np.triu(sum(v**2 for v in v2))
        flat = np.argsort(-strength2, axis=None)[:top]
        pairs = [tuple(int(x) for x in np.unravel_index(f, strength2.shape)) for f in flat if strength2.flat[f] > 0]
        pairs += [(m, m) for m in modes if (m, m) not in pairs]
        lines.append(
            f"  {'pair':>11s} {'sym':>7s} {'meV':>15s} " + " ".join(f"{'V2' + k[1:]:>11s}" for k in _V_KEYS)
        )
        for i, j in pairs:
            lines.append(
                f"  {f'{i + 1}_{j + 1}':>11s} {f'{syms[i]}/{syms[j]}':>7s} {f'{freqs[i]:.2f}/{freqs[j]:.2f}':>15s} "
                + " ".join(f"{v[i, j]:11.4g}" for v in v2)
            )
    return "\n".join(lines)
