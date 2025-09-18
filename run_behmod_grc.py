#!/usr/bin/env python3
"""
run_behmod_grc.py
MF→GrC (N_syn=4) runner with SpikeArrays, single-file version.

Folder layout (run from inside biophysical_model/):
  run_behmod_grc.py
  grc_models/
    IaF_GrC.nml
    IaF_GrC_0.50.nml (optional; used if --f-mf matches)
  spikes/
    MF_StatesOnly2.nml
  synapses/
    RothmanMFToGrCAMPA_4.xml
    RothmanMFToGrCNMDA_4.xml
  network_structures/
    GCLconnectivity_4.pkl
  results/
    behmod/  (auto-created)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle as pkl
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from neuroml import NeuroMLDocument, Network, Population, IncludeType, SynapticConnection
from pyneuroml.pynml import read_neuroml2_file, write_neuroml2_file, run_lems_with_jneuroml
from pyneuroml.lems.LEMSSimulation import LEMSSimulation

BASE = Path(__file__).resolve().parent  # always resolve relative to this script


# ---------- helpers ----------
def resolve_in_dir(base_dir: Path, user_value: Optional[str], default_filename: str) -> Path:
    """If user_value is a bare filename, look for it in base_dir; else resolve as given."""
    if user_value is None:
        return (base_dir / default_filename).resolve()
    p = Path(user_value)
    if len(p.parts) == 1:
        return (base_dir / p).resolve()
    return p.resolve()


def load_connectivity_checked(path: Path) -> Tuple[np.ndarray, int, int, str]:
    with open(path, "rb") as f:
        payload = pkl.load(f)
    if "conn_mat" not in payload:
        raise ValueError(f"{path} missing 'conn_mat'")
    conn_mat = np.asarray(payload["conn_mat"])
    N_mf, N_grc = conn_mat.shape
    fanin = conn_mat.sum(axis=0)
    if not np.all(fanin == 4):
        raise ValueError(f"{path}: requires N_syn=4; unique fan-ins={np.unique(fanin)}")
    sha12 = hashlib.sha256(conn_mat.astype(np.uint8).tobytes()).hexdigest()[:12]
    conn_id = f"{path.stem}__{sha12}"
    return conn_mat, int(N_mf), int(N_grc), conn_id


def ensure_run_dir(out_root: Path, conn_id: str, f_mf: Optional[float], rep: int, label: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    fmf = f"{f_mf:.2f}" if f_mf is not None else "NA"
    run_dir = out_root / "conn" / conn_id / f"fmf_{fmf}" / f"rep_{rep:03d}" / f"{ts}_{label}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def resolve_grc(grc_dir: Path, grc_override: Optional[str], f_mf: Optional[float]) -> Tuple[Path, str]:
    if grc_override:
        p = Path(grc_override).resolve()
        if not p.exists():
            raise FileNotFoundError(f"--grc not found: {p}")
        return p, "explicit"
    if f_mf is not None:
        cand = (grc_dir / f"IaF_GrC_{f_mf:.2f}.nml").resolve()
        if cand.exists():
            return cand, "per_f_mf"
    default = (grc_dir / "IaF_GrC.nml").resolve()
    if not default.exists():
        raise FileNotFoundError(f"Missing default GrC model: {default}")
    return default, "default"


# ---------- build & run (single-file) ----------
def build_and_maybe_run(
    connectivity: Path,
    spikes: Path,
    grc_dir: Path,
    syn_dir: Path,
    outdir_root: Path,
    label: str,
    duration: float,
    dt: float,
    f_mf: Optional[float],
    rep: int,
    run_flag: bool,
    grc_override: Optional[str],
    max_memory: str,
):
    # 1) inputs
    conn_mat, N_mf, N_grc, conn_id = load_connectivity_checked(connectivity)
    grc_path, grc_sel_mode = resolve_grc(grc_dir, grc_override, f_mf)

    ampa_xml = (syn_dir / "RothmanMFToGrCAMPA_4.xml").resolve()
    nmda_xml = (syn_dir / "RothmanMFToGrCNMDA_4.xml").resolve()

    for p in [spikes, grc_path, ampa_xml, nmda_xml]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    # 2) read SpikeArrays (ids + count check)
    spike_doc = read_neuroml2_file(str(spikes))
    spike_arrays = getattr(spike_doc, "spike_arrays", None)
    if not spike_arrays:
        raise ValueError(f"{spikes} contains no SpikeArray elements.")
    mf_ids = [sa.id for sa in spike_arrays]
    if len(mf_ids) != N_mf:
        raise AssertionError(f"SpikeArrays: expected {N_mf}, found {len(mf_ids)}")

    # 3) run folder and copy includes
    run_dir = ensure_run_dir(outdir_root, conn_id, f_mf, rep, label)
    spikes_local = run_dir / spikes.name
    grc_local    = run_dir / grc_path.name
    ampa_local   = run_dir / ampa_xml.name
    nmda_local   = run_dir / nmda_xml.name
    shutil.copy2(spikes, spikes_local)
    shutil.copy2(grc_path, grc_local)
    shutil.copy2(ampa_xml, ampa_local)
    shutil.copy2(nmda_xml, nmda_local)

    # 4) build NeuroML network (relative includes)
    net_id = "MF_GrC_Network"
    doc = NeuroMLDocument(id=f"MF_GrC_{label}")
    net = Network(id=net_id); doc.networks.append(net)
    doc.includes.append(IncludeType(href=grc_local.name))
    doc.includes.append(IncludeType(href=spikes_local.name))

    # IMPORTANT: update if your GrC component id differs
    grc_cell_id = "IaF_GrC"
    net.populations.append(Population(id="GrCPop", component=grc_cell_id, size=N_grc))
    for sid in mf_ids:
        net.populations.append(Population(id=f"{sid}_pop", component=sid, size=1))

    for mf_idx, sid in enumerate(mf_ids):
        targets = np.where(conn_mat[mf_idx, :] == 1)[0]
        for grc_ix in targets:
            for syn in ["RothmanMFToGrCAMPA", "RothmanMFToGrCNMDA"]:
                net.synaptic_connections.append(
                    SynapticConnection(from_=f"{sid}_pop[0]", to=f"GrCPop[{grc_ix}]", synapse=syn)
                )

    # 5) work INSIDE run_dir for all file ops (Option A)
    cwd = os.getcwd()
    os.chdir(run_dir)
    try:
        net_filename = f"network_{label}.net.nml"
        write_neuroml2_file(doc, net_filename)  # validator finds the relative includes we just copied

        sim = LEMSSimulation(f"Sim_{label}", duration, dt)  # no lems_seed (jNeuroML 0.14 compat)
        sim.assign_simulation_target(net_id)
        sim.include_neuroml2_file(net_filename)
        sim.include_neuroml2_file(grc_local.name)
        sim.include_neuroml2_file(spikes_local.name)
        sim.include_lems_file(ampa_local.name, include_included=False)
        sim.include_lems_file(nmda_local.name, include_included=False)

        sim.create_event_output_file("GrCspikes", f"GrC_spikes_{label}.dat")
        for i in range(N_grc):
            sim.add_selection_to_event_output_file("GrCspikes", i, f"GrCPop[{i}]", "spike")
        sim.create_event_output_file("MFspikes", f"MF_spikes_{label}.dat")
        for mf_idx, sid in enumerate(mf_ids):
            sim.add_selection_to_event_output_file("MFspikes", mf_idx, f"{sid}_pop[0]", "spike")

        lems_file_name = Path(sim.save_to_file()).name  # filename in run_dir

        # 6) metadata (written inside run_dir)
        Path("conn_meta.json").write_text(json.dumps({
            "connectivity_file": str(connectivity),
            "conn_id": conn_id,
            "shape": [N_mf, N_grc],
            "fan_in": 4
        }, indent=2))
        Path("resolved_config.json").write_text(json.dumps({
            "label": label, "duration": duration, "dt": dt, "rep": rep, "f_mf": f_mf,
            "N_mf": N_mf, "N_grc": N_grc,
            "grc_model_path": str(grc_path), "grc_selection_mode": grc_sel_mode,
            "spikes": str(spikes),
            "syn_ampa": str(ampa_xml), "syn_nmda": str(nmda_xml),
            "connectivity_path": str(connectivity),
            "outputs_dir": str(run_dir), "lems_file": lems_file_name
        }, indent=2))

        if not run_flag:
            print(f"[emit-only] prepared run in: {run_dir}")
        else:
            res = run_lems_with_jneuroml(lems_file_name, max_memory=max_memory, nogui=True)
            Path("run_meta.json").write_text(json.dumps({"result": str(res)}, indent=2))
            print(f"✅ completed; outputs in {run_dir}")
    finally:
        os.chdir(cwd)


# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="MF→GrC (N_syn=4) per-run folder builder/runner (single file).")

    # we always run from inside biophysical_model/
    ap.add_argument("--connectivity", default="network_structures/GCLconnectivity_4.pkl",
                    help="Connectivity .pkl (path relative to biophysical_model/)")
    ap.add_argument("--spikes", default="spikes/MF_StatesOnly2.nml",
                    help="SpikeArray .nml (path relative to biophysical_model/)")
    ap.add_argument("--grc-dir", default="grc_models", help="Folder with IaF_GrC*.nml")
    ap.add_argument("--syn-dir", default="synapses", help="Folder with synapse XMLs")
    ap.add_argument("--outdir", default="results/behmod", help="Root output folder (inside biophysical_model/)")

    ap.add_argument("--f-mf", dest="f_mf", type=float,
                    help="If IaF_GrC_{f_mf:.2f}.nml exists in grc_models/, it will be used")
    ap.add_argument("--grc", dest="grc_override",
                    help="Explicit GrC .nml (overrides auto selection)")
    ap.add_argument("--rep", type=int, default=1)
    ap.add_argument("--duration", type=float, default=40000)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--label", default="behmod")

    ap.add_argument("--emit-only", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--max-memory", default="8G")

    args = ap.parse_args()
    if not args.emit_only and not args.run:
        args.run = True

    # resolve everything relative to biophysical_model/
    conn_dir    = (BASE / "network_structures").resolve()
    spikes_dir  = (BASE / "spikes").resolve()
    grc_dir     = (BASE / args.grc_dir).resolve() if not Path(args.grc_dir).is_absolute() else Path(args.grc_dir)
    syn_dir     = (BASE / args.syn_dir).resolve() if not Path(args.syn_dir).is_absolute() else Path(args.syn_dir)
    outdir_root = (BASE / args.outdir).resolve() if not Path(args.outdir).is_absolute() else Path(args.outdir)

    connectivity = resolve_in_dir(conn_dir,   args.connectivity, "GCLconnectivity_4.pkl")
    spikes       = resolve_in_dir(spikes_dir, args.spikes,       "MF_StatesOnly2.nml")

    try:
        build_and_maybe_run(
            connectivity=connectivity,
            spikes=spikes,
            grc_dir=grc_dir,
            syn_dir=syn_dir,
            outdir_root=outdir_root,
            label=args.label,
            duration=args.duration,
            dt=args.dt,
            f_mf=args.f_mf,
            rep=args.rep,
            run_flag=args.run,
            grc_override=args.grc_override,
            max_memory=args.max_memory,
        )
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
