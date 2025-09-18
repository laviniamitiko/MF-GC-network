#!/usr/bin/env python3
# run_behmod_grc_flat.py
# Single-folder version: write & run everything in CWD with relative includes.

import argparse, json, pickle as pkl
from pathlib import Path
import numpy as np
from datetime import datetime

from neuroml import NeuroMLDocument, Network, Population, IncludeType, SynapticConnection
from pyneuroml.pynml import read_neuroml2_file, write_neuroml2_file, run_lems_with_jneuroml
from pyneuroml.lems.LEMSSimulation import LEMSSimulation

def load_connectivity_checked(path: Path):
    with open(path, "rb") as f:
        payload = pkl.load(f)
    conn_mat = payload["conn_mat"]
    if not isinstance(conn_mat, np.ndarray):
        conn_mat = np.asarray(conn_mat)
    N_mf, N_grc = conn_mat.shape
    fanin = np.sum(conn_mat, axis=0)
    if not np.all(fanin == 4):
        raise ValueError(f"{path}: requires N_syn=4; unique fan-ins={np.unique(fanin)}")
    return conn_mat, int(N_mf), int(N_grc)

def main():
    ap = argparse.ArgumentParser(description="MF→GrC (N_syn=4) single-folder runner with SpikeArrays.")
    ap.add_argument("--connectivity", required=True, help="GCLconnectivity_4*.pkl (path)")
    ap.add_argument("--spikes", required=True, help="SpikeArray .nml in this folder (e.g., MF_StatesOnly2.nml)")
    ap.add_argument("--grc", default="IaF_GrC.nml", help="GrC model .nml in this folder (default: IaF_GrC.nml)")
    ap.add_argument("--label", default="states_2", help="label for filenames")
    ap.add_argument("--duration", type=float, default=40000.0, help="ms")
    ap.add_argument("--dt", type=float, default=0.1, help="ms")
    ap.add_argument("--max-memory", default="8G")
    ap.add_argument("--emit-only", action="store_true")
    ap.add_argument("--run", action="store_true")
    args = ap.parse_args()
    if not args.emit_only and not args.run:
        args.run = True

    cwd = Path(".").resolve()
    conn_path = Path(args.connectivity).resolve()
    spikes_name = Path(args.spikes).name      # filenames only (must exist in CWD)
    grc_name = Path(args.grc).name            # filenames only (must exist in CWD)

    # sanity: check required files in CWD
    for fname in [spikes_name, grc_name, "RothmanMFToGrCAMPA_4.xml", "RothmanMFToGrCNMDA_4.xml"]:
        if not (cwd / fname).exists():
            raise FileNotFoundError(f"Expected {fname} in {cwd}; not found.")

    # load connectivity & spikes
    conn_mat, N_mf, N_grc = load_connectivity_checked(conn_path)
    spike_doc = read_neuroml2_file(spikes_name)   # relative read; file is in CWD
    spike_arrays = getattr(spike_doc, "spike_arrays", None)
    if not spike_arrays:
        raise ValueError(f"{spikes_name} contains no SpikeArray elements.")
    mf_ids = [sa.id for sa in spike_arrays]
    if len(mf_ids) != N_mf:
        raise AssertionError(f"Expected {N_mf} spike arrays, found {len(mf_ids)} in {spikes_name}")

    # build NeuroML network (relative includes)
    net_id = "MF_GrC_Network"
    doc = NeuroMLDocument(id=f"MF_GrC_{args.label}")
    net = Network(id=net_id)
    doc.networks.append(net)
    doc.includes.append(IncludeType(href=grc_name))
    doc.includes.append(IncludeType(href=spikes_name))

    # IMPORTANT: component id must match the one defined in grc_name; often "IaF_GrC"
    # If your file defines a different id, change below accordingly.
    grc_cell_id = "IaF_GrC"

    net.populations.append(Population(id="GrCPop", component=grc_cell_id, size=N_grc))
    for sid in mf_ids:
        net.populations.append(Population(id=f"{sid}_pop", component=sid, size=1))

    for mf_idx, sid in enumerate(mf_ids):
        targets = np.where(conn_mat[mf_idx, :] == 1)[0]
        for grc_ix in targets:
            for syn in ["RothmanMFToGrCAMPA", "RothmanMFToGrCNMDA"]:
                net.synaptic_connections.append(SynapticConnection(
                    from_=f"{sid}_pop[0]", to=f"GrCPop[{grc_ix}]", synapse=syn
                ))

    net_file = cwd / f"network_{args.label}.net.nml"
    write_neuroml2_file(doc, str(net_file))

    # build LEMS (all includes by filename; LEMS will live in CWD)
    sim = LEMSSimulation(f"Sim_{args.label}", args.duration, args.dt)  # no lems_seed
    sim.assign_simulation_target(net_id)

    sim.include_neuroml2_file(net_file.name)
    sim.include_neuroml2_file(grc_name)
    sim.include_neuroml2_file(spikes_name)
    sim.include_lems_file("RothmanMFToGrCAMPA_4.xml", include_included=False)
    sim.include_lems_file("RothmanMFToGrCNMDA_4.xml", include_included=False)

    # outputs (filename only -> written in CWD)
    sim.create_event_output_file("GrCspikes", f"GrC_spikes_{args.label}.dat")
    for i in range(N_grc):
        sim.add_selection_to_event_output_file("GrCspikes", i, f"GrCPop[{i}]", "spike")

    sim.create_event_output_file("MFspikes", f"MF_spikes_{args.label}.dat")
    for mf_idx, sid in enumerate(mf_ids):
        sim.add_selection_to_event_output_file("MFspikes", mf_idx, f"{sid}_pop[0]", "spike")

    lems_file = Path(sim.save_to_file()).name  # ensure we refer to filename in CWD

    # write a tiny metadata json for your records (optional)
    meta = {
        "connectivity_path": str(conn_path),
        "spikes": spikes_name,
        "grc_model": grc_name,
        "syn_ampa": "RothmanMFToGrCAMPA_4.xml",
        "syn_nmda": "RothmanMFToGrCNMDA_4.xml",
        "N_mf": N_mf, "N_grc": N_grc,
        "dt": args.dt, "duration": args.duration,
        "label": args.label,
        "generated_at": datetime.now().isoformat(timespec="seconds")
    }
    (cwd / f"resolved_config_{args.label}.json").write_text(json.dumps(meta, indent=2))

    if args.emit_only:
        print(f"[emit-only] wrote {net_file.name} and {lems_file} in {cwd}")
        return

    res = run_lems_with_jneuroml(lems_file, max_memory=args.max_memory, nogui=True)
    (cwd / f"run_meta_{args.label}.json").write_text(json.dumps({"result": str(res)}, indent=2))
    print(f"✅ completed; outputs: MF_spikes_{args.label}.dat, GrC_spikes_{args.label}.dat")

if __name__ == "__main__":
    main()
