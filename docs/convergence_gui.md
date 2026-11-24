# Interactive convergence GUI — design

`spinlabs gui` is a local desktop viewer for the convergence packs made by
`spinlabs package --conv`. It reuses the same analysis as
`spinlabs plot convergence` and adds the spin-phonon coupling channels as a
second view.

## Scope

- One window, Tkinter + embedded Matplotlib (`python3-tk` is the only extra
  system requirement; one new pip dependency, `sv-ttk`, for the theme, see
  Appearance).
- Convergence data only in this version. The shell is structured so another
  viewer can become another top-level tab later.
- The registry is read-only. `spinlabs package --conv` remains the only way to
  register data. Unregistered `.npz` files and decont folders can be opened for
  the session through File → Open or `spinlabs gui <path>`.

## Data and tabs

- A top-level tab is a **defect group**: defect + cell size + displacement
  convention. Different cells or displacement conventions get separate tabs,
  matching the separate summary figures the CLI writes.
- Inside one tab the Summary has one column per selected **scheme** (AO, UO,
  full decont, or a non-standard folder key), as in the CLI summary figure. A defect with only a decont pack shows one column; NV 512 with AO and
  UO selected shows two.
- The registry panel lists defect groups. Selecting one reveals the available
  scheme folders with checkboxes and a per-scheme date dropdown (newest first).
  Opening a group creates a tab, or focuses the tab already open for that group.
  Changing the scheme selection updates that tab's columns.
- Inside a tab the date dropdown of a column switches to another registered
  date of the same scheme, across its date folders, in place.
- On launch the previous session's tabs are restored when their files still
  exist; otherwise only the newest defect group opens, with all of its latest
  available schemes checked.
- Pack tabs are closable (Ctrl+W, close others/all). Per-tab state includes
  visible modes, active mode, active column, scheme checkboxes, Norm/Channels
  choice and advanced parameters; registry filters are global.

## Views

Each tab has nested **Summary** and **Mode detail** views.

- **Summary**: two stacked rows (first and second order) and one column per
  selected scheme. Log |Q| axis ticked at the run step sizes.
- **Mode detail**: the CLI's 3 × 2 grid — principal components with linear fit,
  principal components with quadratic fit and residuals, finite differences per
  traceless part. One column at a time, chosen by the tab's active-column
  selector; the active mode is selected by clicking a row in the mode list.
- The mode list shows mode number, symmetry and frequency when a registered
  phonon file provides them. Checkboxes choose which modes appear in Summary;
  All / None / Invert and a text filter are provided. The active mode for Mode
  detail is independent of the checkbox state.  View settings, the mode list
  and the collapsible Advanced settings share one scrollable sidebar tab.
- Changing a setting builds the replacement figure while the current one stays
  on screen, then swaps canvases once the new figure has been drawn. A watch
  cursor is shown while the rebuild runs, so the plot does not blink empty and
  the window does not repaint an intermediate state.

## Difference schemes and norm view

- Forward and Central are independent checkboxes. When both are checked both
  point sets, fit lines and error curves are drawn; the safe-|Q| shading comes
  from Forward (the production scheme) and Central's windows are listed in the
  report. When Forward is unchecked Central supplies the shading. Central is
  enabled when at least one selected pack has −Q runs; packs without them
  simply have no Central series and say so in the report. The checkboxes also
  filter the Mode-detail finite-difference row; with both checked its CLI
  combined row is kept.
- The **Norm** view uses exactly the current summary's values: first order
  `||D(Q) − D₀|| / Q` and second order `||D(2Q) − 2D(Q) + D₀|| / Q²`, or the
  central equivalents. The norm is over the five traceless tensor parts.
  Column titles use the CLI's `<defect>_<cell> <scheme>` form, and the legends
  and marker shapes match the CLI summary for a single scheme; a distinct open
  marker is used only when Forward and Central are overlaid.
  Safe bands, the common |Q| window and the grey error curve are the current
  norm-based quantities. The GUI extends the CLI's six-colour palette so every
  selected mode can be shown. On screen, central points use distinct open
  markers when both schemes are overlaid; with a single scheme the CLI's row
  marker and fill are kept. The common safe-|Q| region uses one shared green in
  Norm and Coupling-channel views.
- If both schemes are unticked, the difference rows show a placeholder and the
  principal components remain available in Mode detail.

## Coupling-channel view

- The other Summary quantity shows all three spin-phonon coupling channels for
  every mode, in both rows:

  | channel | production name | marker |
  | --- | --- | --- |
  | `V_{Δ0}` | `V_0_0` | circle |
  | `V_{Δ±1}` | `V_0_pm` | triangle |
  | `V_{Δ±2}` | `V_p_m` | square |

- Every mode shows all three channels, including symmetry-forbidden ones, so
  the forbidden channels act as a visual zero check. Forbidden channels are
  drawn but never narrow the safe region.
- Channel values use the production derivative definitions and production V
  formulas, on the convergence x-scale: MHz per unit perturbation scale. (The
  production run's absolute V values are per SI displacement and require the
  phonon spectrum; that conversion is not part of this view.)
- First order at each measured step: `V(D(Q) − D₀) / Q` for forward, and
  `V((D(Q) − D(−Q))/2) / Q` for central.
- Second order at each step: production's diagonal difference
  `d2D = (D(2Q) − 2D(Q) + D₀)/Q²` where `2Q` was run, or
  `(D(Q) + D(−Q) − 2D₀)/Q²`; the channel value is `V(d2D)/2`.
- The horizontal Q → 0 lines come from the same polynomial fits the norm view
  uses: `V(b)` for first order; `V(c)` for second order (equivalent to
  `V(2c)/2`), with the fitted `c` from the pair cubic (forward) or quartic
  (central) fit.
- Channel points are one point per |Q| and channel. Colour = mode; marker
  shape = channel. Forward points are filled, Central points open; Norm mode
  keeps the current filled +Q / open −Q convention.
- Forbidden channels are excluded from the safety aggregation. Unknown mode
  symmetry (no registered phonon file) includes all three conservatively and
  warns.
- Per-channel error and safe window: the existing conservative norm error
  scalars `err1`/`err2` are used for every channel; each channel substitutes its
  own fitted coefficient into the existing SNR/truncation formulas, with the
  next Taylor term obtained by applying the same channel functional to the
  fitted `c`/`d`/`e`. Each allowed channel gets `q_lo..q_hi`; the shaded common
  region is their intersection over modes and allowed channels. Forbidden
  channels, unresolved channels and channels below `weak_fraction` of the
  strongest allowed channel are drawn faint and excluded.
- The horizontal channel lines are drawn for every channel with a positive
  fitted Q → 0 value; an exactly zero coefficient is marked at the log-axis
  floor so the forbidden-channel zero check stays visible. Allowed channels get
  a thicker segment over their own safe window; a single vertical common region
  is shaded. The grey conservative error curve remains. The report table lists
  each channel's Q → 0 value, window and flags.
- The Norm/Channels choice is a Summary-tab control only; Mode detail keeps the
  component view.

## Report and export

- Window geometry, registry filters, the last directory, report visibility and
  the open tabs with their per-tab state are remembered in `~/.spinlabs/gui.json`.

- A collapsible bottom pane has two tabs:
  - **Table** for the current view: mode, symmetry, frequency, coefficients,
    safe windows and flags (one row per mode in Norm, one per mode × channel in
    Channels).
  - **Details**: the CLI-style `format_snr` text for the active mode/column plus
    warnings (missing Q=0, unfinished runs, unknown symmetries, modes skipped).
- Save figure rebuilds the current view at the CLI's canonical figure size and
  layout under Matplotlib's default style, not the GUI's muted on-screen theme,
  and writes PNG, SVG or PDF through the Save figure menu, with CLI-like names
  under `figures/convergence/`.  This keeps exported axes, ticks, borders,
  legend frames and font sizes identical to `spinlabs plot convergence`.
  Export CSV writes the current table. The status bar shows the written path.
- Advanced parameters live in a collapsed sidebar section: noise override
  (automatic pooled sigma by default), SNR target, truncation tolerance,
  weak-mode fraction, drop the Q=0 anchor (Mode detail), deviation plot and
  quadratic fit (Mode detail; the quadratic extra values appear in Details).
  Defaults match the CLI.

## Failure behaviour

- A pack without a Q=0 anchor is shown with a warning badge, can be opened, and
  Summary explains that it cannot measure differences; Mode detail still shows
  the principal components.
- Unfinished runs (NaN tensors) are dropped with a report warning, as the CLI
  does.
- Too few amplitudes for the SNR estimate: the mode is listed as skipped in the
  Details pane and omitted from the summary row, matching the CLI. The channel
  view still draws its raw points and marks the fit as unavailable.
- Phonon symmetry and frequency follow the main pipeline's pairing: a pack's
  recorded full spectrum (`phonon_file`, plus its `aligned` tag) first, then the
  registered full spectra, then `phonon_data.npz` / `phonon_data_aligned.npz`
  next to the pack or under the stored ground-state defect folder.  A
  symmetry-reduced file is resolved to its full spectrum.  Packs without an
  `aligned` tag prefer the aligned full spectrum when both are present; the
  Advanced tab can force Auto, Aligned full or Unaligned full per tab.
  `package --conv` now copies and registers the full spectrum alongside the
  convergence pack, using the aligned file when the trees displaced along
  aligned modes.  If no full
  spectrum covers the highest mode, the table leaves symmetry/frequency blank
  and Details says which files were rejected and why.
- An empty registry shows a short empty state with the `spinlabs package
  --conv` hint and the File → Open fallback.

## Appearance

The window uses the Sun Valley ttk theme (the Windows 11-style `sv-ttk`), with a
dark header, white cards on a soft grey surface, one blue accent and a
consistent type scale.  `src/spinlabs/gui/theme.py` owns the small custom layer
and falls back to the previous dependency-free clam theme if `sv-ttk` is not
installed.  `sv-ttk` is a declared dependency; reinstall with
`pip install -e .` after pulling this change.  Matplotlib's on-screen figures
use matching surface colours, muted axes and typography.
