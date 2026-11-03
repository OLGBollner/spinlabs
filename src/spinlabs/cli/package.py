"""CLI command for packaging raw ZFS simulation data into .npz files.

The raw folders are scanned first and the available data is printed; the
user then confirms before anything is written (skip with --yes, or stop after
the listing with --scan).
"""

import argparse
import json
import re
import shutil

import numpy as np
from datetime import date
from pathlib import Path
from typing import NamedTuple

from spinlabs.cli import add_verbosity_args
from spinlabs.logging_utils import attach_db_logging, section
from spinlabs.models import RawZFSData, normalize_calc_method
from spinlabs.vasp.trees import tree_displacement
from spinlabs.registry import ensure_db, register_file
from spinlabs.pack_convergence import ConvergenceScan, save_convergence, scan_convergence
from spinlabs.parsers import parse_zfs_simulation_dataset, split_defect_folder_name
from spinlabs.vasp.run_checks import BASIS_SOURCE
from spinlabs.vasp.settings import (
    compare_settings,
    outcar_settings,
    read_incar,
    start_differences,
    summary_line,
    tree_settings,
)
from spinlabs.vasp.slurm_logs import summarize
from spinlabs.vasp.trees import default_phonon
from spinlabs.vasp.verify import unfinished_runs

from spinlabs.logging_utils import get_logger

LOGGER = get_logger(__name__)


def build_package_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--defect-folder",
        type=Path,
        nargs="+",
        metavar="FOLDER",
        help="Root folder(s) for one defect each, e.g. ../NV_512 ../ClV_128. With --conv: exactly one folder containing runs/.",
    )
    parser.add_argument(
        "--register-only",
        type=Path,
        metavar="DATA_PATH",
        default=None,
        help="Don't package anything: scan DATA_PATH for packaged .npz files and add them to the registry.",
    )
    parser.add_argument(
        "--conv",
        action="store_true",
        help="Package convergence runs (runs/<mode>_pert_<p>/OUTCAR) into one npz instead of raw simulation folders.",
    )
    parser.add_argument(
        "--ground-state",
        type=Path,
        nargs="?",
        const=_GS_AUTO,
        default=None,
        metavar="OUTCAR",
        help=(
            "With --conv: ground-state OUTCAR; its ZFS tensor is stored as 'd0' (the Q=0 point) in the npz. "
            "Flag without a path: auto-discover <defect-folder>/ZFS_hyp/OUTCAR (all) or ZFS_occup/OUTCAR (occup)."
        ),
    )
    parser.add_argument(
        "--orders",
        type=str,
        default="first,second",
        help="Comma-separated orders to package: first, second (default: both).",
    )
    parser.add_argument(
        "--pert",
        type=float,
        nargs="+",
        default=None,
        help="Perturbation scale(s) to package. Default: every pert_* folder found.",
    )
    parser.add_argument(
        "--central",
        action="store_true",
        help="Central differences: package each first_order/pert_<p> with its first_order/pert_-<p> runs "
        "(stored as first_order_minus), for (D(Q) - D(-Q))/2Q and (D(Q) + D(-Q) - 2 D0)/Q^2. Second-order "
        "trees are not packaged with them (central and forward differences cannot be mixed).",
    )
    parser.add_argument(
        "--method",
        type=str,
        nargs="+",
        choices=["all", "approx"],
        default=["all", "approx"],
        metavar="METHOD",
        help="ZFS calculation method(s) to package: all and/or approx (default: both).",
    )
    parser.add_argument(
        "--no-mass-weighted",
        action="store_true",
        help="For trees whose job scripts name no displacement (DISPLACEMENT or MASS_WEIGHTED): their "
        "structures were displaced by Q*e instead of Q*e/sqrt(m) (runs made before mass weighting was added). "
        "Stored as displacement=unweighted; the analysis then corrects the amplitudes.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data"),
        help="Root folder for packaged data (default: ./data).",
    )

    parser.add_argument(
        "--scan",
        action="store_true",
        help="Only scan: parse the simulation folders and print the available data, without asking to save.",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Save the scanned data without asking for confirmation (for non-interactive use).",
    )

    add_verbosity_args(parser)

    parser.set_defaults(func=_run)


_ORDER_DIRS = {"first": "first_order", "second": "second_order"}
_ORDER_NUM = {"first": 1, "second": 2}
_GS_AUTO = Path("__auto__")  # --ground-state given without a path: auto-discover


def _run(args: argparse.Namespace) -> None:
    attach_db_logging(ensure_db())
    if getattr(args, "register_only", None) is not None:
        _register_only(args.register_only)
        return

    if not args.defect_folder:
        raise SystemExit("--defect-folder is required (unless --register-only is given)")
    defect_folders = [f.resolve() for f in args.defect_folder]
    for f in defect_folders:
        if not f.is_dir():
            raise SystemExit(f"Defect folder not found: {f}")

    if getattr(args, "conv", False):
        if len(defect_folders) > 1:
            raise SystemExit("--conv takes exactly one --defect-folder; got " + str(len(defect_folders)))
        _run_conv(args, defect_folders[0])
        return

    scanned: list[_Scanned] = []
    phonons: dict[tuple[str, int], Path] = {}
    for defect_folder in defect_folders:
        found = _scan_raw(args, defect_folder, args.method)
        scanned.extend(found)
        phonon = _find_phonon_npz(defect_folder)
        if found and phonon is not None:
            phonons[(found[0].defect, found[0].cell_size)] = phonon
    section(LOGGER, "Summary")
    _print_scan_summary(scanned, phonons)
    _print_mass_weighting(scanned, cli_unweighted=getattr(args, "no_mass_weighted", False))
    if args.scan or not scanned:
        return
    if not getattr(args, "yes", False) and not _confirm("Save the perturbation data to file? (Y/n): "):
        LOGGER.info("nothing saved")
        return
    section(LOGGER, "Save perturbation data")
    _save_raw(scanned, args.output_root, cli_unweighted=getattr(args, "no_mass_weighted", False))
    _save_phonons(phonons, args.output_root)


class _Scanned(NamedTuple):
    """One parsed perturbation scale, ready to be saved."""

    defect: str
    cell_size: int
    calc_method: str
    pert_scale: float
    raw: RawZFSData
    # Displacement convention of the trees' job scripts (DISPLACEMENTS), None when they do not say
    displacement: str | None = None


_DISPLACEMENT_TEXT = {
    "normalized": "step * u/|u| (normalized: same Cartesian step per mode)",
    "mass_weighted": "Q*e/sqrt(m) (mass-weighted)",
    "unweighted": "Q*e (not mass-weighted)",
}


def _displacement_of(scanned: _Scanned, cli_unweighted: bool) -> str:
    """The convention a dataset is saved with: its trees' setting, else the --no-mass-weighted flag
    (trees whose scripts say nothing predate both settings and were mass-weighted)."""
    if scanned.displacement is not None:
        return scanned.displacement
    return "unweighted" if cli_unweighted else "mass_weighted"


def _print_mass_weighting(scanned: list[_Scanned], cli_unweighted: bool) -> None:
    """Which displacement convention each dataset is saved with, and any conflict with the flag."""
    if not scanned:
        return
    print("Displacements:")
    for s in scanned:
        name = f"{s.defect}_{s.cell_size} {s.calc_method} pert {s.pert_scale:g}"
        convention = _displacement_of(s, cli_unweighted)
        source = "trees' job scripts" if s.displacement is not None else (
            "--no-mass-weighted" if cli_unweighted else "default")
        print(f"  {name}: {_DISPLACEMENT_TEXT[convention]}  [{source}]")
        if s.displacement not in (None, "unweighted") and cli_unweighted:
            print(f"    !! --no-mass-weighted ignored: the trees' job scripts say {s.displacement}")


def _confirm(prompt: str) -> bool:
    """Ask a yes/no question; Enter means yes, EOF (no terminal) means no."""
    try:
        answer = input(prompt).strip().lower()
    except EOFError:
        print()
        return False
    return answer in ("", "y", "yes")


def _scan_raw(args: argparse.Namespace, defect_folder: Path, methods: list[str]) -> list[_Scanned]:
    """Parse every requested order/method/pert under one defect folder."""
    orders = [o.strip().lower() for o in args.orders.split(",") if o.strip()]
    unknown = [o for o in orders if o not in _ORDER_DIRS]
    if unknown:
        raise SystemExit(f"Unknown order(s): {', '.join(unknown)} (expected 'first' and/or 'second')")
    central = getattr(args, "central", False)
    if central:
        if "first" not in orders:
            raise SystemExit("--central packages first-order runs at +p and -p; --orders must include first")
        if "second" in orders:
            LOGGER.info("--central: second-order trees are not packaged (central and forward differences cannot mix)")
        orders = ["first"]

    defect, cell_size = split_defect_folder_name(defect_folder.name)
    if cell_size is None:
        raise SystemExit(f"Cell size not found in defect folder name: {defect_folder.name} (expected e.g. NV_512)")

    section(LOGGER, f"Scan {defect_folder.name}: {defect_folder}")

    def _collect(order: str) -> dict[float, Path]:
        """Map pert_scale -> pert folder for one order, filtered to the requested scales."""
        order_root = defect_folder / _ORDER_DIRS[order]
        if not order_root.is_dir():
            LOGGER.warning(f"{defect_folder.name}: no {order} order folder ({order_root}); {order} order skipped")
            return {}
        perts = {}
        for pert_dir in sorted(order_root.glob("pert_*")):
            pert_scale = float(pert_dir.name.split("_")[1])
            if args.pert is not None and pert_scale not in args.pert:
                continue
            if central and pert_scale < 0:  # the -p partner of a +p dataset, not one of its own
                continue
            perts[pert_scale] = pert_dir
        return perts

    def _minus_folder(pert_scale: float) -> Path | None:
        """The first_order/pert_-<p> folder holding the -Q runs of step pert_scale."""
        for pert_dir in sorted((defect_folder / _ORDER_DIRS["first"]).glob("pert_-*")):
            if np.isclose(float(pert_dir.name.split("_")[1]), -pert_scale):
                return pert_dir
        return None

    found: list[_Scanned] = []
    for method in methods:
        calc_method = normalize_calc_method(method)
        datasets = {o: _collect(o) for o in orders}
        for o in orders:
            scales = " ".join(f"{p:g}" for p in datasets[o])
            LOGGER.info(f"{calc_method} {o} order: {len(datasets[o])} perturbation scale(s) {scales}".rstrip())
        pert_scales = sorted(set().union(*datasets.values()))
        if args.pert is not None:
            missing = [p for p in args.pert if p not in pert_scales]
            for p in missing:
                LOGGER.warning(f"{defect_folder.name}: --pert {p:g} not found in any order folder; skipped")
        if not pert_scales:
            LOGGER.warning(f"{defect_folder.name} {calc_method}: no perturbation data found; skipped")
            continue
        ground = _ground_state_settings(defect_folder, calc_method)
        for pert_scale in pert_scales:
            raw = None
            trees = {}
            weighting: set[bool] = set()
            for o in orders:
                if pert_scale not in datasets[o]:
                    continue
                data = parse_zfs_simulation_dataset(
                    sim_folder=datasets[o][pert_scale], order=_ORDER_NUM[o], calc_method=method, verbose=True
                )
                raw = data if raw is None else raw.combine(data)
                basis = datasets[o][pert_scale] / calc_method
                if basis.is_dir():
                    trees[_ORDER_DIRS[o]] = tree_settings(basis)
                    unfinished = unfinished_runs(basis)
                    if unfinished is not None:
                        trees[_ORDER_DIRS[o]]["unfinished"] = unfinished
                    convention = tree_displacement(basis)
                    if convention is not None:
                        weighting.add(convention)
            if central and raw is not None:
                minus_dir = _minus_folder(pert_scale)
                if minus_dir is None:
                    LOGGER.warning(
                        f"{defect_folder.name} {calc_method} pert {pert_scale:g}: --central but no "
                        f"{_ORDER_DIRS['first']}/pert_-{pert_scale:g} folder with the -Q runs; skipped"
                    )
                    continue
                minus = parse_zfs_simulation_dataset(sim_folder=minus_dir, order=1, calc_method=method, verbose=True)
                raw = raw.with_minus(minus)
                basis = minus_dir / calc_method
                if basis.is_dir():
                    trees["first_order_minus"] = tree_settings(basis)
                    unfinished = unfinished_runs(basis)
                    if unfinished is not None:
                        trees["first_order_minus"]["unfinished"] = unfinished
                    convention = tree_displacement(basis)
                    if convention is not None:
                        weighting.add(convention)
            if raw is None or not (raw.first_order or raw.second_order):
                LOGGER.warning(f"{defect_folder.name} {calc_method} pert {pert_scale:g}: no ZFS data parsed; skipped")
                continue
            raw.vasp_settings = _vasp_settings(ground, trees)
            if len(weighting) > 1:
                LOGGER.warning(
                    f"{defect_folder.name} {calc_method} pert {pert_scale:g}: first and second order trees use "
                    f"different displacements ({', '.join(sorted(weighting))}); they are not consistent"
                )
            convention = weighting.pop() if len(weighting) == 1 else None
            found.append(_Scanned(defect, cell_size, calc_method, pert_scale, raw, convention))
    return found


def _ground_state_settings(defect_folder: Path, calc_method: str) -> dict[str, str] | None:
    """VASP settings of the ground-state ZFS run (<defect>/ZFS_hyp or ZFS_occup), or None if absent."""
    return _run_folder_settings(defect_folder / BASIS_SOURCE[calc_method])


def _run_folder_settings(folder: Path) -> dict[str, str] | None:
    """VASP settings of one run folder from its INCAR, else its OUTCAR; None if it has neither."""
    if (folder / "INCAR").is_file():
        return read_incar(folder / "INCAR")
    if (folder / "OUTCAR").is_file():
        return outcar_settings(folder / "OUTCAR")
    return None


# Full-spectrum phonon files, most preferred first. Symmetry-reduced spectra
# (``*sym*``) are skipped: packaged mode indices refer to the full spectrum.
_PHONON_NAMES = ("phonon_data.npz", "phonons.npz")
PACKAGED_PHONON_NAME = "phonon_data.npz"


def _find_phonon_npz(defect_folder: Path) -> Path | None:
    """The full-spectrum phonon .npz of a defect, or None (with a warning).

    Looks in <defect>/data/ (where the VASP trees keep it) before the defect folder itself.
    """
    search_dirs = [default_phonon(defect_folder).parent, defect_folder]
    for folder in search_dirs:
        for name in _PHONON_NAMES:
            if (folder / name).is_file():
                return folder / name
    candidates = sorted(p for folder in search_dirs for p in folder.glob("*phonon*.npz"))
    full = [p for p in candidates if "sym" not in p.stem]
    if len(full) == 1:
        return full[0]
    if full:
        LOGGER.warning(
            f"{defect_folder.name}: several phonon .npz files ({', '.join(p.name for p in full)}); "
            f"rename the full spectrum to {PACKAGED_PHONON_NAME} to package it"
        )
    elif candidates:
        LOGGER.warning(
            f"{defect_folder.name}: only symmetry-reduced phonon .npz "
            f"({', '.join(p.name for p in candidates)}); phonon data not packaged"
        )
    else:
        LOGGER.warning(f"{defect_folder.name}: no phonon .npz found; phonon data not packaged")
    return None


def _save_phonons(phonons: dict[tuple[str, int], Path], output_root: Path) -> None:
    """Copy each phonon file to <output_root>/<defect>_<cell>/phonon_data.npz and register it."""
    for (defect, cell_size), src in phonons.items():
        dest = output_root / f"{defect}_{cell_size}" / PACKAGED_PHONON_NAME
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        register_file(ensure_db(), dest, "phonon", defect=defect, cell_size=cell_size)
        LOGGER.info(f"saved phonon data: {dest} (copied from {src})")


def _print_scan_summary(scanned: list[_Scanned], phonons: dict[tuple[str, int], Path] | None = None) -> None:
    if not scanned:
        print("No perturbation data found.")
        return
    header = f"{'defect':<12} {'method':<20} {'pert':>8} {'1st-order modes':>16} {'2nd-order pairs':>16}"
    print("Available perturbation data:")
    print(header)
    print("-" * len(header))
    for s in scanned:
        # central: the 2nd-order column counts the modes with their -Q run
        second = f"{len(s.raw.first_order_minus)} (-Q)" if s.raw.is_central else str(len(s.raw.second_order))
        print(
            f"{f'{s.defect}_{s.cell_size}':<12} {s.calc_method:<20} {s.pert_scale:>8g} "
            f"{len(s.raw.first_order):>16} {second:>16}"
        )
    if phonons:
        print("\nPhonon data:")
        for (defect, cell_size), path in phonons.items():
            print(f"{f'{defect}_{cell_size}':<12} {path}")
    _print_settings(scanned)
    if phonons:
        _print_shape_flags(scanned, phonons)
    print()


def _print_settings(scanned: list[_Scanned]) -> None:
    """Ground-state vs run VASP settings per dataset; every difference is a warning."""
    print("\nVASP settings (ground state vs the runs differenced against it):")
    for s in scanned:
        _print_settings_of(f"{s.defect}_{s.cell_size} {s.calc_method} pert {s.pert_scale:g}", s.raw.vasp_settings)


def _vasp_settings(ground: dict[str, str] | None, trees: dict[str, dict]) -> dict:
    """The vasp_settings record of one dataset: ground state, trees and their differences."""
    return {
        "ground_state": ground,
        "trees": trees,
        "problems": compare_settings(ground, trees),
        "start_differences": start_differences(ground, trees),
    }


def _print_settings_of(name: str, settings: dict | None) -> None:
    """One dataset's line of the VASP settings check (see _vasp_settings), with its differences."""
    settings = settings or {}
    ground, trees = settings.get("ground_state"), settings.get("trees", {})
    n_runs = sum(t["n_runs"] for t in trees.values())
    if not n_runs:
        print(f"  {name}: no run INCAR or OUTCAR found; not checked")
        return
    where = "" if ground is not None else " (no ground-state INCAR/OUTCAR found)"
    problems = settings.get("problems", [])
    start = settings.get("start_differences", [])
    status = "consistent" if not problems else f"{len(problems)} difference(s)"
    print(f"  {name}: {status}; {summary_line(ground, trees)}{where}")
    for problem in problems:
        print(f"    !! {problem}")
        LOGGER.debug(f"{name}: settings: {problem}")  # kept in the DB log; printed once above
    for label, tree in trees.items():
        unfinished = tree.get("unfinished") or {}
        if not unfinished:
            continue
        print(f"    !! {label}: {len(unfinished)} run(s) not finished (from the tree's slurm-*.out logs):")
        for line in summarize(unfinished):
            print(f"         {line}")
            LOGGER.debug(f"{name}: {label} not finished: {line}")
    if start:
        print(f"    starting point only (ISTART/ICHARG, same converged result): {len(start)} difference(s)")
        for line in start:
            LOGGER.debug(f"{name}: starting point: {line}")


def _print_shape_flags(scanned: list[_Scanned], phonons: dict[tuple[str, int], Path]) -> None:
    """Flag entries whose ZFS change has the wrong shape for their modes' symmetry."""
    from spinlabs.parsers import parse_phonon_data
    from spinlabs.vasp.run_checks import dataset_shape_flags

    print("\nD-tensor shape check (A1/A2 modes may only change the A1 part, E modes mainly the E part):")
    spectra = {}
    for key, path in phonons.items():
        try:
            spectra[key] = parse_phonon_data(path)
        except Exception as error:  # a diagnostic must never stop packaging
            print(f"  {key[0]}_{key[1]}: cannot read {path} ({error.__class__.__name__}); shape check skipped")
    for s in scanned:
        spectrum = spectra.get((s.defect, s.cell_size))
        if spectrum is None:
            continue
        name = f"{s.defect}_{s.cell_size} {s.calc_method} pert {s.pert_scale:g}"
        try:
            flags = dataset_shape_flags(s.raw, spectrum)
        except Exception as error:
            print(f"  {name}: shape check failed ({error.__class__.__name__}: {error}); skipped")
            continue
        if not flags:
            print(f"  {name}: all entries OK")
            continue
        shown = ", ".join(folder for folder, _ in flags[:15]) + (", ..." if len(flags) > 15 else "")
        print(f"  {name}: {len(flags)} suspicious run folder(s): {shown}")
        for folder, problem in flags:
            LOGGER.debug(f"{name}: runs/{folder}: {problem}")  # per-run reasons with -v
        if len(flags) > 15:
            print("    (every folder and its reason: rerun with -v)")


def _save_raw(scanned: list[_Scanned], output_root: Path, cli_unweighted: bool = False) -> None:
    """Write each scanned dataset to <output_root>/<defect>_<cell>/pert_<p>/<method>/<date>/ and register it."""
    date_dir = date.today().strftime("%Y%m%d")
    for s in scanned:
        s.raw.displacement = _displacement_of(s, cli_unweighted)
        s.raw.mass_weighted = s.raw.displacement != "unweighted"
        pert_folder = output_root / f"{s.defect}_{s.cell_size}" / f"pert_{s.pert_scale:g}" / s.calc_method / date_dir
        pert_folder.mkdir(parents=True, exist_ok=True)
        save_path = s.raw.save(pert_folder / s.raw._default_name())
        register_file(
            ensure_db(),
            save_path,
            "packaged_zfs",
            defect=s.defect,
            cell_size=s.cell_size,
            calc_method=s.calc_method,
            pert_order=([1] if s.raw.has_first_order else []) + ([2] if s.raw.has_second_order else []),
            pert_scale=s.pert_scale,
        )
        LOGGER.info(f"saved raw ZFS data: {save_path}")


_DICT_ORDER_SUFFIXES = {
    "12d": [1, 2],
    "1d": [1],
    "2d": [2],
}


def _dict_nonempty(data: np.lib.npyio.NpzFile, key: str) -> bool:
    """True when the npz holds a non-empty pickled dict under ``key``."""
    if key not in data.files:
        return False
    val = data[key]
    if val is None:
        return False
    if isinstance(val, np.ndarray) and val.shape == () and val.dtype == object:
        val = val.item()
    try:
        return len(val) > 0
    except TypeError:
        return False


def _register_only(data_path: Path) -> None:
    """Register already-packaged .npz files under data_path without re-packaging."""
    data_path = data_path.resolve()
    if not data_path.is_dir():
        raise SystemExit(f"Data path not found: {data_path}")

    section(LOGGER, f"Register packaged files: {data_path}")
    conn = ensure_db()
    registered, skipped = 0, 0
    for npz_path in sorted(data_path.rglob("*.npz")):
        if npz_path.name == PACKAGED_PHONON_NAME:
            defect, cell_size = split_defect_folder_name(npz_path.parent.name)
            if cell_size is None:
                LOGGER.warning(f"phonon file outside a <defect>_<cell> folder, not registered: {npz_path}")
                skipped += 1
                continue
            register_file(conn, npz_path, "phonon", defect=defect, cell_size=cell_size)
            LOGGER.debug(f"registered phonon data ({defect}_{cell_size}): {npz_path}")
            registered += 1
            continue
        try:
            with np.load(str(npz_path), allow_pickle=True) as data:
                keys = set(data.files)
                if _CONV_KEYS.issubset(keys):
                    if _register_conv(conn, npz_path, data):
                        registered += 1
                    else:
                        skipped += 1
                    continue
                required = {f.name for f in RawZFSData.SCHEMA if not f.optional} | {
                    "defect",
                    "cell_size",
                    "order",
                }
                if not required.issubset(keys):
                    LOGGER.warning(f"not a packaged ZFS, convergence or phonon file, not registered: {npz_path}")
                    skipped += 1
                    continue
                defect = str(data["defect"])
                cell_size = int(data["cell_size"])
                pert_scale = float(data["pert_scale"])
                calc_method = normalize_calc_method(str(data["calc_method"]))
                # Derive orders from the npz content itself: a non-empty
                # first_order / second_order mapping means that perturbation
                # order is present in the file. Fall back to the filename
                # suffix only when the order keys are missing or unreadable.
                if _dict_nonempty(data, "first_order") or _dict_nonempty(data, "second_order"):
                    # central packages hold their second order as the -Q runs of first_order_minus
                    second = _dict_nonempty(data, "second_order") or _dict_nonempty(data, "first_order_minus")
                    orders = ([1] if _dict_nonempty(data, "first_order") else []) + ([2] if second else [])
                else:
                    suffix_orders = _DICT_ORDER_SUFFIXES.get(npz_path.stem.rsplit("_", 1)[-1])
                    if suffix_orders is None:
                        LOGGER.warning(
                            f"no order data and no recognizable file name suffix, not registered: {npz_path}"
                        )
                        skipped += 1
                        continue
                    orders = suffix_orders
        except Exception as exc:  # unreadable npz: report and move on
            LOGGER.warning(f"cannot read {npz_path} ({exc}); not registered")
            skipped += 1
            continue
        register_file(
            conn,
            npz_path,
            "packaged_zfs",
            defect=defect,
            cell_size=cell_size,
            calc_method=calc_method,
            pert_order=orders,
            pert_scale=pert_scale,
        )
        LOGGER.debug(
            f"registered packaged ZFS data ({defect}_{cell_size} {calc_method} order {orders} pert {pert_scale:g}): "
            f"{npz_path}"
        )
        registered += 1
    LOGGER.info(f"registered {registered} file(s), skipped {skipped}")


_CONV_KEYS = {"modes", "perts", "tensors"}  # what save_convergence writes besides metadata


def _register_conv(conn, npz_path: Path, data) -> bool:
    """Register a convergence npz ('package --conv'); False (with a warning) when it cannot be placed.

    Defect, cell and method come from its metadata, else from the file name
    <defect>_<cell>_<date>_conv_<folder suffix>.npz that _run_conv writes.
    """
    try:
        meta = json.loads(str(data["metadata"])) if "metadata" in data.files else {}
    except (TypeError, ValueError):
        meta = {}
    name_part, _, suffix = npz_path.stem.partition("_conv_")
    name_defect, name_cell = split_defect_folder_name(name_part)
    defect = meta.get("defect") or name_defect
    cell = meta.get("cell") or name_cell
    if not defect or cell is None:
        LOGGER.warning(f"convergence file without defect/cell in its metadata or name, not registered: {npz_path}")
        return False
    method = meta.get("method") or _conv_method(meta.get("method_key") or suffix)
    calc_method = normalize_calc_method(method) if method else None
    register_file(conn, npz_path, "convergence", defect=defect, cell_size=int(cell), calc_method=calc_method)
    LOGGER.debug(f"registered convergence data ({defect}_{cell} {calc_method}): {npz_path}")
    return True


def _conv_method(suffix: str) -> str | None:
    """'all' or 'approx' from a convergence folder suffix, None when it says neither.

    test_all / test_occup, or AE (all electrons) / UE (unpaired electrons) as a token,
    e.g. AE_mass_weighted.
    """
    tokens = {t.lower() for t in re.split(r"[_.\-]", suffix)}
    if "all" in suffix or "ae" in tokens:
        return "all"
    if "occup" in suffix or "ue" in tokens:
        return "approx"
    return None


def _auto_ground_state(defect_folder: Path, method: str | None) -> Path | None:
    """Flag-only --ground-state: pick ZFS_hyp (all) or ZFS_occup (occup) OUTCAR."""
    if method == "all":
        zfs_folder = "ZFS_hyp"
    elif method in ("occup", "approx"):
        zfs_folder = "ZFS_occup"
    else:
        LOGGER.warning(
            f"{defect_folder.name}: convergence folder name says neither 'all'/'AE' nor 'occup'/'UE'; "
            "d0 not stored (pass --ground-state OUTCAR)"
        )
        return None
    candidate = defect_folder / zfs_folder / "OUTCAR"
    if candidate.exists():
        LOGGER.debug(f"ground state: {candidate}")
        return candidate
    LOGGER.warning(f"no ground-state OUTCAR at {candidate}; d0 not stored (pass --ground-state OUTCAR)")
    return None


def _run_conv(args: argparse.Namespace, defect_folder: Path) -> None:
    # _run_conv warns when a --conv run passes options that only apply to raw
    # packaging. These defaults MUST mirror the argparse defaults below; if
    # one changes, change both, or a matching user value silently skips the
    # warning.
    defaults = {"orders": "first,second", "pert": None, "method": ["all", "approx"]}
    for opt in ("orders", "pert", "method"):
        if getattr(args, opt, defaults[opt]) != defaults[opt]:
            LOGGER.warning(f"--{opt.replace('_', '-')} is ignored with --conv")
    if getattr(args, "no_mass_weighted", False):
        LOGGER.warning("--no-mass-weighted is ignored with --conv")
    conv_roots = sorted(p for p in defect_folder.glob("convergence*") if p.is_dir())
    if not conv_roots:
        raise SystemExit(f"No convergence* folders found under {defect_folder}")
    scanned = [s for s in (_scan_conv(args, defect_folder, root) for root in conv_roots) if s is not None]
    section(LOGGER, "Summary")
    _print_conv_summary(scanned)
    if getattr(args, "scan", False) or not scanned:
        return
    if not getattr(args, "yes", False) and not _confirm("Save the convergence data to file? (Y/n): "):
        LOGGER.info("nothing saved")
        return
    section(LOGGER, "Save convergence data")
    date_dir = date.today().strftime("%Y%m%d")
    for s in scanned:
        leaf = args.output_root / f"{s.defect}_{s.cell}" / (s.calc_method or "conv") / date_dir
        leaf.mkdir(parents=True, exist_ok=True)
        output = leaf / f"{s.defect}_{s.cell}_{date_dir}_conv_{s.suffix}.npz"
        save_convergence(
            s.scan, output, ground_state=s.ground_state, defect=s.defect, cell=s.cell, method=s.method,
            method_key=s.suffix,
        )
        register_file(ensure_db(), output, "convergence", defect=s.defect, cell_size=s.cell, calc_method=s.calc_method)


class _ConvScanned(NamedTuple):
    """One parsed convergence folder, ready to be saved."""

    defect: str
    cell: int
    suffix: str  # folder name without 'convergence', e.g. 'test_all'; names the output file
    method: str | None  # 'all' or 'approx' from the suffix, None when it says neither
    calc_method: str | None
    ground_state: Path | None  # OUTCAR stored as d0
    scan: ConvergenceScan
    vasp_settings: dict


def _scan_conv(args: argparse.Namespace, defect_folder: Path, root: Path) -> _ConvScanned | None:
    """Parse one convergence* folder and check its settings against the ground state; None if it has no runs."""
    section(LOGGER, f"Scan convergence folder {defect_folder.name}/{root.name}")
    suffix = re.sub(r"^convergence[._-]*", "", root.name).lstrip("_-") or "conv"
    method = _conv_method(suffix)
    calc_method = normalize_calc_method(method) if method else None
    defect, cell = split_defect_folder_name(defect_folder.name)
    try:
        scan = scan_convergence(root)
    except ValueError as exc:
        LOGGER.warning(f"{defect_folder.name}/{root.name}: {exc}; not packaged")
        return None
    LOGGER.info(f"{scan.n_runs} run folder(s), {scan.n_finished} with a ZFS tensor")
    ground_state = args.ground_state
    if ground_state == _GS_AUTO:
        ground_state = _auto_ground_state(defect_folder, method)
    # Settings are checked against the stored d0's run, else the method's ZFS_hyp/ZFS_occup.
    if ground_state is not None:
        ground = _run_folder_settings(Path(ground_state).parent)
    elif calc_method in BASIS_SOURCE:
        ground = _ground_state_settings(defect_folder, calc_method)
    else:
        ground = None
    settings = _vasp_settings(ground, {"runs": tree_settings(root)})
    return _ConvScanned(defect, cell, suffix, method, calc_method, ground_state, scan, settings)


def _format_steps(perts: list[float]) -> str:
    """'±0.025 ±0.05' when every size was run at both signs, else the steps as found.

    Q = 0 runs (<mode>_pert_0) are the unperturbed point, not a step, and are left out.
    """
    perts = [p for p in perts if p != 0]
    sizes = sorted({abs(p) for p in perts})
    if all(s in perts and -s in perts for s in sizes):
        return " ".join(f"±{s:g}" for s in sizes)
    return " ".join(f"{p:g}" for p in perts)


def _print_conv_summary(scanned: list[_ConvScanned]) -> None:
    if not scanned:
        print("No convergence data found.")
        return
    columns = ("defect", "folder", "method", "modes", "steps", "finished")
    rows = [
        (
            f"{s.defect}_{s.cell}", s.suffix, s.calc_method or "-", " ".join(map(str, s.scan.modes)),
            _format_steps(s.scan.perts), f"{s.scan.n_finished}/{s.scan.n_grid}",
        )
        for s in scanned
    ]
    widths = [max(len(c), *(len(r[i]) for r in rows)) for i, c in enumerate(columns)]
    header = " ".join(c.ljust(w) for c, w in zip(columns, widths)).rstrip()
    print("Available convergence data:")
    print(header)
    print("-" * len(header))
    for row in rows:
        print(" ".join(v.ljust(w) for v, w in zip(row, widths)).rstrip())
    for s in scanned:
        gaps = s.scan.n_grid - s.scan.n_runs
        if s.scan.unfinished:
            shown = ", ".join(s.scan.unfinished[:10]) + (", ..." if len(s.scan.unfinished) > 10 else "")
            print(f"  !! {s.suffix}: {len(s.scan.unfinished)} run(s) without a ZFS tensor: {shown}")
        if gaps:
            print(f"  !! {s.suffix}: {gaps} mode/step combination(s) have no run folder")

    print("\nGround state (stored as d0):")
    for s in scanned:
        print(f"  {s.suffix}: {s.ground_state or 'none (d0 not stored)'}")

    print("\nDisplacements:")
    for s in scanned:
        convention = s.scan.displacement or "mass_weighted"
        source = "trees' job scripts" if s.scan.displacement is not None else "default"
        print(f"  {s.defect}_{s.cell} {s.suffix}: {_DISPLACEMENT_TEXT[convention]}  [{source}]")

    print("\nVASP settings (ground state vs the convergence runs):")
    for s in scanned:
        _print_settings_of(f"{s.defect}_{s.cell} {s.suffix}", s.vasp_settings)
