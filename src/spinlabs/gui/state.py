"""Small JSON persistence for the convergence GUI (window, filters, tabs)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path

from spinlabs.convergence import SNR_TARGET, TRUNC_TOL
from spinlabs.convergence_summary import WEAK_FRACTION

DEFAULT_GEOMETRY = "1500x950"


def default_state_path() -> Path:
    return Path(os.path.expanduser("~")) / ".spinlabs" / "gui.json"


@dataclass
class TabState:
    group_key: str
    columns: list[dict] = field(default_factory=list)
    visible_modes: list[int] | None = None  # None means "not recorded", [] means "none visible"
    active_mode: int | None = None
    active_column: int = 0
    forward: bool = True
    central: bool = False
    display: str = "norm"
    phonon: str = "auto"
    noise: float | None = None
    target: float = SNR_TARGET
    tol: float = TRUNC_TOL
    weak_fraction: float = WEAK_FRACTION
    drop_anchor: bool = False
    deviation: bool = False
    quad: bool = False


@dataclass
class GuiState:
    geometry: str = DEFAULT_GEOMETRY
    filters: dict = field(default_factory=dict)
    report_visible: bool = True
    active_tab: str | None = None
    tabs: list[TabState] = field(default_factory=list)
    last_directory: str | None = None
    # comparison tabs (spinlabs.gui.comparison.ComparisonTab.state): entries, view options and settings
    comparisons: list[dict] = field(default_factory=list)


def load_state(path: Path | None = None) -> GuiState:
    """Read the saved state; a missing or corrupt file is a clean default."""
    path = Path(path) if path is not None else default_state_path()
    try:
        raw = json.loads(path.read_text())
    except (OSError, ValueError, TypeError):
        return GuiState()
    if not isinstance(raw, dict):
        return GuiState()
    tabs = []
    for item in raw.get("tabs") or []:
        if not isinstance(item, dict) or not item.get("group_key"):
            continue
        known = {key: item[key] for key in TabState.__dataclass_fields__ if key in item}
        try:
            tabs.append(TabState(**known))
        except TypeError:
            continue
    filters = raw.get("filters") if isinstance(raw.get("filters"), dict) else {}
    return GuiState(
        geometry=str(raw.get("geometry") or DEFAULT_GEOMETRY),
        filters=filters,
        report_visible=bool(raw.get("report_visible", True)),
        active_tab=raw.get("active_tab"),
        tabs=tabs,
        last_directory=raw.get("last_directory"),
        comparisons=[c for c in raw.get("comparisons") or [] if isinstance(c, dict) and c.get("key")],
    )


def save_state(state: GuiState, path: Path | None = None) -> Path:
    """Write the state atomically; returns the path written."""
    path = Path(path) if path is not None else default_state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(state)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)
    return path
