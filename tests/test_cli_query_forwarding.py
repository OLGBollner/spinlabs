"""`spinlabs run --query` must honour every option it accepts.

Before this was fixed, ``_run_query`` forwarded only 5 of the ~15 arguments the
non-query path forwards, so a command like::

    spinlabs run --query --t-end 500 --t-step 100 --init-state ms_1 --run-name mine

exited 0 having silently used the default temperature grid, ``ms_0`` and a
timestamped run directory.
"""

import argparse

import pytest

import spinlabs.cli.run as cli_run
import spinlabs.registry as registry


def _parse(argv):
    parser = argparse.ArgumentParser()
    cli_run.build_run_parser(parser)
    return parser.parse_args(argv)


@pytest.fixture
def captured(monkeypatch):
    """Run the query path against a stubbed registry, capturing pipeline kwargs."""
    calls = []

    def fake_pipeline(**kwargs):
        calls.append(kwargs)
        return {}

    monkeypatch.setattr(cli_run, "run_full_pipeline", fake_pipeline)
    monkeypatch.setattr(registry, "ensure_db", lambda *a, **k: object())
    return calls


def _matches(monkeypatch, *paths):
    monkeypatch.setattr(registry, "find_packaged", lambda *a, **k: list(paths))


def test_query_forwards_temperature_grid_and_init_state(captured, monkeypatch):
    _matches(monkeypatch, "/data/NV_512_raw_zfs_data_all_bands_12d.npz")
    args = _parse(
        [
            "--query",
            "--t-start", "0",
            "--t-end", "500",
            "--t-step", "100",
            "--init-state", "ms_1",
        ]
    )
    args.func(args)

    assert len(captured) == 1
    kwargs = captured[0]
    assert (kwargs["t_start"], kwargs["t_end"], kwargs["t_step"]) == (0.0, 500.0, 100.0)
    assert kwargs["init_state"] == ["ms_1"]


def test_query_forwards_explicit_temperature_grid(captured, monkeypatch):
    _matches(monkeypatch, "/data/NV_512_raw_zfs_data_all_bands_12d.npz")
    args = _parse(["--query", "--temperatures", "10", "20", "30"])
    args.func(args)

    assert captured[0]["temperatures"] == [10.0, 20.0, 30.0]


def test_query_forwards_run_name_for_a_single_match(captured, monkeypatch):
    _matches(monkeypatch, "/data/NV_512_raw_zfs_data_all_bands_12d.npz")
    args = _parse(["--query", "--run-name", "my_named_run"])
    args.func(args)

    assert captured[0]["run_name"] == "my_named_run"


def test_query_runs_every_match_under_its_own_run_name(captured, monkeypatch):
    _matches(
        monkeypatch,
        "/data/NV_512_raw_zfs_data_all_bands_12d.npz",
        "/data/ClV_128_raw_zfs_data_all_bands_12d.npz",
    )
    args = _parse(["--query", "--run-name", "my_named_run"])
    args.func(args)

    assert [c["run_name"] for c in captured] == [
        "my_named_run_NV_512_raw_zfs_data_all_bands_12d",
        "my_named_run_ClV_128_raw_zfs_data_all_bands_12d",
    ]


def test_query_forwards_order_and_pert_scale(captured, monkeypatch):
    _matches(monkeypatch, "/data/NV_512_raw_zfs_data_all_bands_12d.npz")
    args = _parse(["--query", "--order", "1", "--pert-scale", "0.025"])
    args.func(args)

    assert captured[0]["order"] == [1]
    assert captured[0]["pert_scale"] == 0.025


def test_query_forwards_defect_and_cell_size(captured, monkeypatch):
    _matches(monkeypatch, "/data/NV_512_raw_zfs_data_all_bands_12d.npz")
    args = _parse(["--query", "--defect", "NV", "--cell-size", "512"])
    args.func(args)

    assert captured[0]["defect"] == "NV"
    assert captured[0]["cell_size"] == 512


def test_query_honours_explicit_phonon_file(captured, monkeypatch):
    """Omitted, the registry resolves the phonon; given, the user's choice wins."""
    _matches(monkeypatch, "/data/NV_512_raw_zfs_data_all_bands_12d.npz")
    args = _parse(["--query", "-ph", "/data/phonopy.yaml", "--two-phonon", "/data/raman.npz"])
    args.func(args)

    assert captured[0]["phonon_file"] == "/data/phonopy.yaml"
    assert captured[0]["two_phonon_file"] == "/data/raman.npz"


def test_query_defaults_to_no_phonon_override(captured, monkeypatch):
    _matches(monkeypatch, "/data/NV_512_raw_zfs_data_all_bands_12d.npz")
    args = _parse(["--query"])
    args.func(args)

    assert captured[0]["phonon_file"] is None


def test_query_respects_the_plot_flag(captured, monkeypatch):
    _matches(monkeypatch, "/data/NV_512_raw_zfs_data_all_bands_12d.npz")

    args = _parse(["--query"])
    args.func(args)
    assert captured[0]["save_plots"] is False

    captured.clear()
    args = _parse(["--query", "--plot"])
    args.func(args)
    assert captured[0]["save_plots"] is True


def test_query_still_infers_method_per_dataset(captured, monkeypatch):
    """Each matched file keeps its own method, which is why --method is not forwarded."""
    _matches(
        monkeypatch,
        "/data/NV_512_raw_zfs_data_all_bands_12d.npz",
        "/data/NV_512_raw_zfs_data_defect_band_approx_12d.npz",
    )
    args = _parse(["--query", "--t-end", "500"])
    args.func(args)

    assert [c["calc_method"] for c in captured] == ["all_bands", "defect_band_approx"]
    assert [c["zfs_folder"] for c in captured] == ["ZFS_hyp", "ZFS_occup"]
    # ... and the forwarded options apply to every dataset, not just the first
    assert {c["t_end"] for c in captured} == {500.0}


def test_query_forwards_everything_the_direct_path_does(captured, monkeypatch):
    """Guard against a new run option being added to _run but forgotten in _run_query."""
    _matches(monkeypatch, "/data/NV_512_raw_zfs_data_all_bands_12d.npz")
    args = _parse(["--query"])
    args.func(args)
    query_kwargs = set(captured[0])

    captured.clear()
    args = _parse(["--raw-zfs-file", "/data/NV_512_raw_zfs_data_all_bands_12d.npz"])
    args.func(args)
    direct_kwargs = set(captured[0])

    # the input-mode arguments are the only ones --query legitimately decides itself
    input_mode = {"sim_folder", "raw_zfs_file", "raw_zfs_file_1d", "raw_zfs_file_2d", "coupling_file"}
    assert direct_kwargs - query_kwargs <= input_mode


def test_query_solves_every_init_state_in_one_run_by_default(captured, monkeypatch):
    _matches(monkeypatch, "/data/NV_512_raw_zfs_data_all_bands_12d.npz")
    args = _parse(["--query", "--run-name", "mine"])
    args.func(args)

    assert len(captured) == 1
    assert captured[0]["init_state"] == ["ms_1", "ms_0", "ms_-1"]
    assert captured[0]["run_name"] == "mine"


def test_non_query_run_solves_every_init_state_in_one_run(captured):
    args = _parse(["--coupling-file", "/data/NV_512_coupling_all_bands.npz", "--method", "all"])
    args.func(args)

    assert len(captured) == 1
    assert captured[0]["init_state"] == ["ms_1", "ms_0", "ms_-1"]


def test_query_accepts_multiple_defects(captured, monkeypatch):
    """`--defect NV DiV ClV` queries each defect and forwards the matching one."""
    seen = []

    def fake_find(conn, *, defect, **kwargs):
        seen.append(defect)
        return [f"/data/{defect}_128_raw_zfs_data_all_bands_1d.npz"]

    monkeypatch.setattr(registry, "find_packaged", fake_find)
    args = _parse(["--query", "--defect", "NV", "DiV", "ClV"])
    args.func(args)

    assert seen == ["NV", "DiV", "ClV"]
    assert [c["defect"] for c in captured] == ["NV", "DiV", "ClV"]


def test_multiple_defects_rejected_without_query():
    args = _parse(["--raw-zfs-file", "x_all_bands.npz", "--defect", "NV", "ClV"])
    with pytest.raises(SystemExit):
        args.func(args)


def test_query_runs_every_filter_combination(captured, monkeypatch):
    """`--defect NV ClV --pert-scale 0.1 0.025` queries all four combinations."""
    seen = []

    def fake_find(conn, *, defect, pert_scale, **kwargs):
        seen.append((defect, pert_scale))
        return [f"/data/{defect}_128_pert_{pert_scale}_all_bands_1d.npz"]

    monkeypatch.setattr(registry, "find_packaged", fake_find)
    args = _parse(["--query", "--defect", "NV", "ClV", "--pert-scale", "0.1", "0.025"])
    args.func(args)

    combos = [("NV", 0.1), ("NV", 0.025), ("ClV", 0.1), ("ClV", 0.025)]
    assert seen == combos
    assert [(c["defect"], c["pert_scale"]) for c in captured] == combos


def test_query_runs_a_dataset_matched_twice_once(captured, monkeypatch):
    monkeypatch.setattr(registry, "find_packaged", lambda *a, **k: ["/data/NV_128_all_bands_1d.npz"])
    args = _parse(["--query", "--cell-size", "128", "512"])
    args.func(args)
    assert len(captured) == 1


@pytest.mark.parametrize("flag", [["--cell-size", "64", "128"], ["--method", "all", "approx"], ["--pert-scale", "0.1", "0.025"]])
def test_multiple_filter_values_rejected_without_query(flag):
    args = _parse(["--raw-zfs-file", "x_all_bands.npz", *flag])
    with pytest.raises(SystemExit):
        args.func(args)


def test_query_continues_past_a_failing_dataset_and_reports_it(monkeypatch, caplog):
    paths = [f"/data/NV_{c}_raw_zfs_data_all_bands_12d.npz" for c in (64, 216, 512)]
    calls = []

    def fake_pipeline(**kwargs):
        calls.append(kwargs["raw_zfs_file"])
        if "216" in kwargs["raw_zfs_file"]:
            raise ValueError("No perturbation tensors in the input")
        return {}

    monkeypatch.setattr(cli_run, "run_full_pipeline", fake_pipeline)
    monkeypatch.setattr(registry, "ensure_db", lambda *a, **k: object())
    _matches(monkeypatch, *paths)
    args = _parse(["--query"])

    with caplog.at_level("INFO", logger="spinlabs"), pytest.raises(
        SystemExit, match=r"1 dataset\(s\) failed:\n  /data/NV_216_raw_zfs_data_all_bands_12d.npz"
    ):
        args.func(args)

    assert calls == paths  # the dataset after the failure still ran
    assert "Dataset 2/3: /data/NV_216_raw_zfs_data_all_bands_12d.npz" in caplog.text
    assert "dataset 2/3 failed: /data/NV_216" in caplog.text
    assert "analyzed 2 of 3 dataset(s)" in caplog.text


def test_query_forwards_sigma(captured, monkeypatch):
    _matches(monkeypatch, "/data/NV_512_raw_zfs_data_all_bands_12d.npz")
    args = _parse(["--query", "--sigma", "2"])
    args.func(args)
    assert captured[0]["sigma"] == 2.0
