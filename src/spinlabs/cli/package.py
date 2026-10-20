"""CLI command for packaging raw ZFS simulation data into .npz files.

The raw folders are scanned first and the available data is printed; the
user then confirms before anything is written (skip with --yes, or stop after
the listing with --scan).
"""

import argparse
import re
import shutil
import warnings

import numpy as np
from datetime import date
from pathlib import Path
from typing import NamedTuple

from spinlabs.cli import add_verbosity_args
from spinlabs.logging_utils import attach_db_logging
from spinlabs.models import RawZFSData, normalize_calc_method
from spinlabs.registry import ensure_db, register_file
from spinlabs.pack_convergence import pack_convergence
from spinlabs.parsers import parse_zfs_simulation_dataset, split_defect_folder_name
from spinlabs.vasp.trees import default_phonon

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
        "--method",
        type=str,
        nargs="+",
        choices=["all", "approx"],
        default=["all", "approx"],
        metavar="METHOD",
        help="ZFS calculation method(s) to package: all or approx (default: all).",
    )
    parser.add_argument(
        "--no-mass-weighted",
        action="store_true",
        help="The perturbed structures were displaced by Q*e instead of Q*e/sqrt(m) (runs made before mass "
        "weighting was added). Stored as mass_weighted=False; the analysis then corrects the amplitudes.",
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
    _print_scan_summary(scanned, phonons)
    if getattr(args, "no_mass_weighted", False) and scanned:
        print("Perturbations marked as NOT mass-weighted (--no-mass-weighted).\n")
    if args.scan or not scanned:
        return
    if not getattr(args, "yes", False) and not _confirm("Save the perturbation data to file? (Y/n): "):
        print("Nothing saved.")
        return
    _save_raw(scanned, args.output_root, mass_weighted=not getattr(args, "no_mass_weighted", False))
    _save_phonons(phonons, args.output_root)


class _Scanned(NamedTuple):
    """One parsed perturbation scale, ready to be saved."""

    defect: str
    cell_size: int
    calc_method: str
    pert_scale: float
    raw: RawZFSData


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

    defect, cell_size = split_defect_folder_name(defect_folder.name)
    if cell_size is None:
        raise SystemExit(f"Cell size not found in defect folder name: {defect_folder.name} (expected e.g. NV_512)")

    LOGGER.info(f"Defect folder: {defect_folder}")

    def _collect(order: str) -> dict[float, Path]:
        """Map pert_scale -> pert folder for one order, filtered to the requested scales."""
        order_root = defect_folder / _ORDER_DIRS[order]
        if not order_root.is_dir():
            warnings.warn(f"Order folder not found, skipping: {order_root}")
            return {}
        perts = {}
        for pert_dir in sorted(order_root.glob("pert_*")):
            pert_scale = float(pert_dir.name.split("_")[1])
            if args.pert is not None and pert_scale not in args.pert:
                continue
            perts[pert_scale] = pert_dir
        return perts

    found: list[_Scanned] = []
    for method in methods:
        calc_method = normalize_calc_method(method)
        datasets = {o: _collect(o) for o in orders}
        for o in orders:
            LOGGER.info(f"  {o} order: {len(datasets[o])} perturbation length(s) found in {defect_folder / _ORDER_DIRS[o]}")
        pert_scales = sorted(set().union(*datasets.values()))
        if args.pert is not None:
            missing = [p for p in args.pert if p not in pert_scales]
            for p in missing:
                warnings.warn(f"Perturbation scale {p:g} not found in {defect_folder.name}; skipping")
        if not pert_scales:
            warnings.warn(f"No perturbation data found in {defect_folder.name} ({calc_method}); skipping")
            continue
        for pert_scale in pert_scales:
            raw = None
            for o in orders:
                if pert_scale not in datasets[o]:
                    continue
                data = parse_zfs_simulation_dataset(
                    sim_folder=datasets[o][pert_scale], order=_ORDER_NUM[o], calc_method=method, verbose=True
                )
                raw = data if raw is None else raw.combine(data)
            if raw is None or not (raw.first_order or raw.second_order):
                warnings.warn(f"No data parsed for {defect_folder.name} ({calc_method}, pert {pert_scale:g}); skipping")
                continue
            found.append(_Scanned(defect, cell_size, calc_method, pert_scale, raw))
    return found


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
        warnings.warn(
            f"Several phonon .npz files in {defect_folder.name} ({', '.join(p.name for p in full)}); "
            f"rename the full spectrum to {PACKAGED_PHONON_NAME} to package it"
        )
    elif candidates:
        warnings.warn(
            f"Only symmetry-reduced phonon .npz in {defect_folder.name} "
            f"({', '.join(p.name for p in candidates)}); phonon data not packaged"
        )
    else:
        warnings.warn(f"No phonon .npz found in {defect_folder.name}; phonon data not packaged")
    return None


def _save_phonons(phonons: dict[tuple[str, int], Path], output_root: Path) -> None:
    """Copy each phonon file to <output_root>/<defect>_<cell>/phonon_data.npz and register it."""
    for (defect, cell_size), src in phonons.items():
        dest = output_root / f"{defect}_{cell_size}" / PACKAGED_PHONON_NAME
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        register_file(ensure_db(), dest, "phonon", defect=defect, cell_size=cell_size)
        LOGGER.info(f"Copied phonon data {src} -> {dest}")


def _print_scan_summary(scanned: list[_Scanned], phonons: dict[tuple[str, int], Path] | None = None) -> None:
    if not scanned:
        print("\nNo perturbation data found.")
        return
    header = f"{'defect':<12} {'method':<20} {'pert':>8} {'1st-order modes':>16} {'2nd-order pairs':>16}"
    print("\nAvailable perturbation data:")
    print(header)
    print("-" * len(header))
    for s in scanned:
        print(
            f"{f'{s.defect}_{s.cell_size}':<12} {s.calc_method:<20} {s.pert_scale:>8g} "
            f"{len(s.raw.first_order):>16} {len(s.raw.second_order):>16}"
        )
    if phonons:
        print("\nPhonon data:")
        for (defect, cell_size), path in phonons.items():
            print(f"{f'{defect}_{cell_size}':<12} {path}")
        _print_shape_flags(scanned, phonons)
    print()


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
            LOGGER.warning(f"{name}: runs/{folder}: {problem}")


def _save_raw(scanned: list[_Scanned], output_root: Path, mass_weighted: bool = True) -> None:
    """Write each scanned dataset to <output_root>/<defect>_<cell>/pert_<p>/<method>/<date>/ and register it."""
    date_dir = date.today().strftime("%Y%m%d")
    for s in scanned:
        s.raw.mass_weighted = mass_weighted
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
        LOGGER.info(f"Saved raw data in: {save_path}")


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

    conn = ensure_db()
    registered, skipped = 0, 0
    for npz_path in sorted(data_path.rglob("*.npz")):
        if npz_path.name == PACKAGED_PHONON_NAME:
            defect, cell_size = split_defect_folder_name(npz_path.parent.name)
            if cell_size is None:
                warnings.warn(f"Phonon file outside a <defect>_<cell> folder, skipping: {npz_path}")
                skipped += 1
                continue
            register_file(conn, npz_path, "phonon", defect=defect, cell_size=cell_size)
            LOGGER.info(f"Registered: {npz_path} (phonon, {defect}_{cell_size})")
            registered += 1
            continue
        try:
            with np.load(str(npz_path), allow_pickle=True) as data:
                keys = set(data.files)
                required = {f.name for f in RawZFSData.SCHEMA if not f.optional} | {
                    "defect",
                    "cell_size",
                    "order",
                }
                if not required.issubset(keys):
                    warnings.warn(f"Not a packaged raw ZFS file, skipping: {npz_path.name}")
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
                    orders = ([1] if _dict_nonempty(data, "first_order") else []) + (
                        [2] if _dict_nonempty(data, "second_order") else []
                    )
                else:
                    suffix_orders = _DICT_ORDER_SUFFIXES.get(npz_path.stem.rsplit("_", 1)[-1])
                    if suffix_orders is None:
                        warnings.warn(
                            f"{npz_path.name}: no order data and no recognizable "
                            "filename suffix, skipping"
                        )
                        skipped += 1
                        continue
                    orders = suffix_orders
        except Exception as exc:  # unreadable npz: report and move on
            warnings.warn(f"Could not read {npz_path.name}: {exc}")
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
        LOGGER.info(f"Registered: {npz_path} ({defect}_{cell_size}, {calc_method}, order {orders}, pert {pert_scale:g})")
        registered += 1
    LOGGER.info(f"Done: {registered} registered, {skipped} skipped")


def _auto_ground_state(defect_folder: Path, method: str | None) -> Path | None:
    """Flag-only --ground-state: pick ZFS_hyp (all) or ZFS_occup (occup) OUTCAR."""
    if method == "all":
        zfs_folder = "ZFS_hyp"
    elif method in ("occup", "approx"):
        zfs_folder = "ZFS_occup"
    else:
        LOGGER.warning(
            "cannot infer ground-state method from folder suffix "
            "(neither 'all' nor 'occup'); skipping ground state",
        )
        return None
    candidate = defect_folder / zfs_folder / "OUTCAR"
    if candidate.exists():
        LOGGER.info(f"ground state: {candidate}")
        return candidate
    LOGGER.warning(f"no ground-state OUTCAR at {candidate}; skipping ground state")
    return None


def _run_conv(args: argparse.Namespace, defect_folder: Path) -> None:
    # _run_conv warns when a --conv run passes options that only apply to raw
    # packaging. These defaults MUST mirror the argparse defaults below; if
    # one changes, change both, or a matching user value silently skips the
    # warning.
    defaults = {"orders": "first,second", "pert": None, "method": ["all"]}
    for opt in ("orders", "pert", "method"):
        if getattr(args, opt, defaults[opt]) != defaults[opt]:
            LOGGER.warning(f"--{opt.replace('_', '-')} is ignored with --conv")
    if getattr(args, "no_mass_weighted", False):
        LOGGER.warning("--no-mass-weighted is ignored with --conv")
    conv_roots = sorted(p for p in defect_folder.glob("convergence*") if p.is_dir())
    if not conv_roots:
        raise SystemExit(f"No convergence* folders found under {defect_folder}")
    method_map = {"all": "all", "occup": "approx"}
    for root in conv_roots:
        LOGGER.info(f"Convergence folder: {root}")
        suffix = re.sub(r"^convergence[._-]*", "", root.name).lstrip("_-") or "conv"
        method = next((v for k, v in method_map.items() if k in suffix), None)
        defect, cell = split_defect_folder_name(defect_folder.name)
        stem = f"{defect}_{cell}_{date.today():%Y%m%d}_conv_{suffix}"
        calc_method = normalize_calc_method(method) if method else method
        date_dir = date.today().strftime("%Y%m%d")
        leaf = args.output_root / f"{defect}_{cell}" / (calc_method or "conv") / date_dir
        leaf.mkdir(parents=True, exist_ok=True)
        output = leaf / f"{stem}.npz"
        ground_state = args.ground_state
        if ground_state == _GS_AUTO:
            ground_state = _auto_ground_state(defect_folder, method)
        pack_convergence(
            root, output, ground_state=ground_state, defect=defect, cell=cell, method=method, method_key=suffix
        )
        register_file(
            ensure_db(), output, "convergence", defect=defect, cell_size=cell, calc_method=calc_method
        )
