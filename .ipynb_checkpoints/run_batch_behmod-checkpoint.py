#!/usr/bin/env python3
"""
run_batch_behmod.py
End-to-end batch pipeline:
  1) generate connectivity (N_syn=4)
  2) generate behavior-modulated MF SpikeArrays for that connectivity
  3) run MF→GrC simulation (via run_behmod_grc.py)

Run from inside biophysical_model/.

New features:
- --manifest results/behmod/batch_manifest.json   # run-by-run records
- --recycle                                       # reuse existing artifacts if inputs match
- --dry-run                                       # print plan only
- --continue-on-error                             # don't abort on a failed run
- --overwrite                                     # force regeneration (ignore recycle)
- --outdir/--grc-dir/--syn-dir passthrough        # forwarded to runner
- per-run stdout/stderr logs

Examples
--------
python run_batch_behmod.py \
  --runs 10 \
  --beh-mat data/191018_13_39_41_dfTrace.mat \
  --minT 100 --maxT 140 --acq-rate 7.2 \
  --pm-frac 0.40 --nm-frac 0.30 \
  --label states_2 --duration 40 --dt 0.1 \
  --base-seed 1000 \
  --f-mf 0.50 \
  --recycle --continue-on-error
"""

from __future__ import annotations
import argparse
import hashlib
import json
import os
import pickle as pkl
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

BASE = Path(__file__).resolve().parent
TOOLS = BASE / "tools"
if TOOLS.exists():
    sys.path.insert(0, str(TOOLS))

# optional imports (prefer callable; fallback to CLI tools)
try:
    from GCL_make_connectivity import generate_connectivity  # type: ignore
except Exception:
    generate_connectivity = None

try:
    # our tool exposes make_spikes_from_behavior; keep a short alias for clarity
    from make_spikes_from_behavior import make_spikes_from_behavior as _make_spikes  # type: ignore
except Exception:
    _make_spikes = None


def sha12_conn(conn_mat: np.ndarray) -> str:
    return hashlib.sha256(conn_mat.astype(np.uint8).tobytes()).hexdigest()[:12]


def run_cmd_capture(cmd: list[str]) -> str:
    """Return stdout (text) or raise CalledProcessError."""
    return subprocess.check_output(cmd, text=True).strip()


def run_cmd_stream(cmd: list[str], stdout_path: Path, stderr_path: Path) -> int:
    """Stream child stdout/stderr to files; return exit code."""
    with stdout_path.open("w") as out, stderr_path.open("w") as err:
        proc = subprocess.Popen(cmd, stdout=out, stderr=err, text=True)
        return proc.wait()


def gen_connectivity(conn_outdir: Path, seed: int, overwrite: bool, recycle: bool) -> Path:
    """
    Generate or reuse a connectivity. Prefer Python callable; fallback to CLI.
    Returns absolute Path to the generated .pkl.
    """
    # If recycling, see if a file named N_grc_*_N_mf_*.json with matching seed exists
    if recycle:
        for meta in sorted(conn_outdir.glob("N_grc_*_N_mf_*.json")):
            try:
                j = json.loads(meta.read_text())
                if j.get("seed") == seed:
                    pkl_path = Path(j["pickle"]).resolve()
                    if pkl_path.exists():
                        return pkl_path
            except Exception:
                continue

    if generate_connectivity is not None:
        p = generate_connectivity(conn_outdir, seed=seed)  # type: ignore
        return Path(p).resolve()

    gen_conn = (TOOLS / "GCL_make_connectivity.py").resolve()
    cmd = [sys.executable, str(gen_conn), "--outdir", str(conn_outdir), "--seed", str(seed)]
    path_str = run_cmd_capture(cmd)
    return Path(path_str).resolve()


def discover_autonamed_spikes(connectivity: Path, beh_mat: Path, minT: float, maxT: float) -> Optional[Path]:
    spikes_dir = (BASE / "spikes").resolve()
    sidecars = sorted(spikes_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for sc in sidecars:
        try:
            meta = json.loads(sc.read_text())
        except Exception:
            continue
        if (Path(meta.get("connectivity", "")).resolve() == connectivity.resolve() and
            Path(meta.get("behavior_mat", "")).resolve() == beh_mat.resolve() and
            float(meta.get("minT", -1)) == float(minT) and
            float(meta.get("maxT", -1)) == float(maxT)):
            out_nml = Path(meta.get("output_nml", ""))
            if out_nml.exists():
                return out_nml.resolve()
    return None


def gen_spikes(
    connectivity: Path,
    beh_mat: Path,
    minT: float,
    maxT: float,
    acq_rate: float,
    pm_frac: float,
    nm_frac: float,
    spike_dt: float,
    label: str,
    out_path: Optional[Path],
    seed_pm: int,
    seed_nm: int,
    seed_ns: int,
    pm_scale: float, pm_offset: float, pm_minrate: float,
    nm_scale: float, nm_offset: float, nm_minrate: float,
    ns_scale: float, ns_offset: float, ns_minrate: float,
    overwrite: bool,
    recycle: bool,
) -> Path:
    """
    Generate or reuse spikes for a given connectivity. Prefer Python callable; fallback to CLI.
    """
    if recycle and out_path is None:
        # try to find the most recent matching spikes auto-named for these inputs
        found = discover_autonamed_spikes(connectivity, beh_mat, minT, maxT)
        if found:
            return found

    if _make_spikes is not None:
        nml_path, _meta = _make_spikes(
            connectivity=str(connectivity),
            beh_mat=str(beh_mat),
            minT=minT, maxT=maxT, acq_rate=acq_rate,
            pm_frac=pm_frac, nm_frac=nm_frac,
            spike_dt=spike_dt, label=label,
            out=str(out_path) if out_path else None,
            pm_scale=pm_scale, pm_offset=pm_offset, pm_minrate=pm_minrate,
            nm_scale=nm_scale, nm_offset=nm_offset, nm_minrate=nm_minrate,
            ns_scale=ns_scale, ns_offset=ns_offset, ns_minrate=ns_minrate,
            seed_pm=seed_pm, seed_nm=seed_nm, seed_ns=seed_ns,
        )
        return Path(nml_path).resolve()

    # CLI fallback
    gen_spk = (TOOLS / "make_spikes_from_behavior.py").resolve()
    cmd = [
        sys.executable, str(gen_spk),
        "--connectivity", str(connectivity),
        "--beh-mat", str(beh_mat),
        "--minT", str(minT), "--maxT", str(maxT),
        "--acq-rate", str(acq_rate),
        "--pm-frac", str(pm_frac), "--nm-frac", str(nm_frac),
        "--spike-dt", str(spike_dt),
        "--label", label,
        "--seed-pm", str(seed_pm), "--seed-nm", str(seed_nm), "--seed-ns", str(seed_ns),
        "--pm-scale", str(pm_scale), "--pm-offset", str(pm_offset), "--pm-minrate", str(pm_minrate),
        "--nm-scale", str(nm_scale), "--nm-offset", str(nm_offset), "--nm-minrate", str(nm_minrate),
        "--ns-scale", str(ns_scale), "--ns-offset", str(ns_offset), "--ns-minrate", str(ns_minrate),
    ]
    if out_path is not None:
        if not overwrite and Path(out_path).exists() and recycle:
            return Path(out_path).resolve()
        cmd += ["--out", str(out_path)]
    print("[INFO] Generating spikes:", " ".join(cmd))
    run_cmd_capture(cmd)

    if out_path is not None:
        return Path(out_path).resolve()

    found = discover_autonamed_spikes(connectivity, beh_mat, minT, maxT)
    if not found:
        raise RuntimeError("Could not locate auto-named spike file; consider passing --spikes-out explicitly.")
    return found


def append_manifest(manifest_path: Path, record: dict) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        data = json.loads(manifest_path.read_text())
        if not isinstance(data, list):  # migrate old format if needed
            data = [data]
    else:
        data = []
    data.append(record)
    manifest_path.write_text(json.dumps(data, indent=2))


def main():
    ap = argparse.ArgumentParser(description="Batch: connectivity → spikes → simulation.")
    ap.add_argument("--runs", type=int, required=True)

    # behavior / spikes
    ap.add_argument("--beh-mat", required=True, help=".mat with 'state_ds'")
    ap.add_argument("--minT", type=float, required=True)
    ap.add_argument("--maxT", type=float, required=True)
    ap.add_argument("--acq-rate", type=float, required=True)
    ap.add_argument("--pm-frac", type=float, required=True)
    ap.add_argument("--nm-frac", type=float, required=True)
    ap.add_argument("--spike-dt", type=float, default=0.001)

    # spike rate params
    ap.add_argument("--pm-scale", type=float, default=50.0)
    ap.add_argument("--pm-offset", type=float, default=7.0)
    ap.add_argument("--pm-minrate", type=float, default=2.0)
    ap.add_argument("--nm-scale", type=float, default=-50.0)
    ap.add_argument("--nm-offset", type=float, default=40.0)
    ap.add_argument("--nm-minrate", type=float, default=2.0)
    ap.add_argument("--ns-scale", type=float, default=0.0)
    ap.add_argument("--ns-offset", type=float, default=7.0)
    ap.add_argument("--ns-minrate", type=float, default=2.0)

    # runner / sim
    ap.add_argument("--label", default="behmod")
    ap.add_argument("--duration", type=float, default=40000)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--f-mf", type=float, help="Use IaF_GrC_{f_mf:.2f}.nml if present; else IaF_GrC.nml")
    ap.add_argument("--grc", help="Explicit GrC .nml to force (overrides --f-mf)")
    ap.add_argument("--rep-offset", type=int, default=0)
    ap.add_argument("--outdir", default="results/behmod")   # pass-through to runner
    ap.add_argument("--grc-dir", default="grc_models")      # pass-through to runner
    ap.add_argument("--syn-dir", default="synapses")        # pass-through to runner

    # seeding
    ap.add_argument("--base-seed", type=int, default=1)

    # toggles / behavior
    ap.add_argument("--emit-only", action="store_true")
    ap.add_argument("--max-memory", default="8G")
    ap.add_argument("--recycle", action="store_true", help="Reuse existing connectivity/spikes when inputs match")
    ap.add_argument("--overwrite", action="store_true", help="Force regeneration (ignore recycle)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--continue-on-error", action="store_true")

    # output / manifest
    ap.add_argument("--spikes-out-dir", default="spikes")
    ap.add_argument("--manifest", default=str(BASE / "results" / "behmod" / "batch_manifest.json"))

    args = ap.parse_args()

    # sanity: fractions
    if args.pm_frac < 0 or args.nm_frac < 0 or args.pm_frac + args.nm_frac > 1.0:
        raise SystemExit("Require pm_frac >=0, nm_frac >=0, pm_frac+nm_frac <= 1.")

    # paths
    conn_dir   = (BASE / "network_structures").resolve()
    spikes_dir = (BASE / args.spikes_out_dir).resolve()
    runner     = (BASE / "run_behmod_grc.py").resolve()
    beh_mat    = Path(args.beh_mat).resolve()
    manifest_path = Path(args.manifest).resolve()

    for p in [conn_dir, spikes_dir]:
        p.mkdir(parents=True, exist_ok=True)
    if not runner.exists():
        raise SystemExit(f"Missing runner: {runner}")
    if not beh_mat.exists():
        raise SystemExit(f"Missing behavior mat: {beh_mat}")

    summary = {"started": datetime.now().isoformat(), "runs_requested": args.runs, "records": []}

    for idx in range(1, args.runs + 1):
        t0 = time.time()
        conn_seed = args.base_seed + idx
        record = {
            "index": idx,
            "conn_seed": conn_seed,
            "status": "planned",
            "started_at": datetime.now().isoformat(),
        }
        try:
            # --- 1) connectivity ---
            connectivity = gen_connectivity(conn_dir, seed=conn_seed, overwrite=args.overwrite, recycle=args.recycle)

            with open(connectivity, "rb") as f:
                payload = pkl.load(f)
            if "conn_mat" not in payload:
                raise RuntimeError(f"{connectivity} missing 'conn_mat'")
            conn_mat = np.asarray(payload["conn_mat"])
            conn_hash = sha12_conn(conn_mat)
            record["connectivity"] = str(connectivity)
            record["N_mf"], record["N_grc"] = int(conn_mat.shape[0]), int(conn_mat.shape[1])
            record["connectivity_hash12"] = conn_hash

            # --- 2) spikes ---
            seed_pm = conn_seed * 7919 + 11
            seed_nm = conn_seed * 6841 + 13
            seed_ns = conn_seed * 4721 + 17
            spikes_out = None  # let tool auto-name; you can set an explicit path if you prefer
            spike_file = gen_spikes(
                connectivity=connectivity,
                beh_mat=beh_mat,
                minT=args.minT, maxT=args.maxT, acq_rate=args.acq_rate,
                pm_frac=args.pm_frac, nm_frac=args.nm_frac,
                spike_dt=args.spike_dt,
                label=args.label,
                out_path=spikes_out,
                seed_pm=seed_pm, seed_nm=seed_nm, seed_ns=seed_ns,
                pm_scale=args.pm_scale, pm_offset=args.pm_offset, pm_minrate=args.pm_minrate,
                nm_scale=args.nm_scale, nm_offset=args.nm_offset, nm_minrate=args.nm_minrate,
                ns_scale=args.ns_scale, ns_offset=args.ns_offset, ns_minrate=args.ns_minrate,
                overwrite=args.overwrite, recycle=args.recycle,
            )
            record["spikes"] = str(spike_file)

            # --- 3) run simulation ---
            rep = args.rep_offset + idx
            cmd = [
                sys.executable, str(runner),
                "--connectivity", str(connectivity),
                "--spikes", str(spike_file),
                "--label", args.label,
                "--duration", str(args.duration),
                "--dt", str(args.dt),
                "--rep", str(rep),
                "--max-memory", str(args.max_memory),
                "--outdir", args.outdir,
                "--grc-dir", args.grc_dir,
                "--syn-dir", args.syn_dir,
            ]
            if args.grc:
                cmd += ["--grc", str(Path(args.grc).resolve())]
            elif args.f_mf is not None:
                cmd += ["--f-mf", f"{args.f_mf:.2f}"]
            cmd += ["--emit-only"] if args.emit_only else ["--run"]
            record["runner_cmd"] = " ".join(cmd)

            if args.dry_run:
                record["status"] = "dry-run"
            else:
                # write per-run logs next to spikes (or in results root if you prefer)
                logs_dir = (BASE / "results" / "behmod" / "logs").resolve()
                logs_dir.mkdir(parents=True, exist_ok=True)
                stdout_path = logs_dir / f"run_{idx:03d}.out"
                stderr_path = logs_dir / f"run_{idx:03d}.err"
                rc = run_cmd_stream(cmd, stdout_path, stderr_path)
                record["stdout"] = str(stdout_path)
                record["stderr"] = str(stderr_path)
                record["return_code"] = rc
                record["status"] = "ok" if rc == 0 else "failed"

            record["elapsed_sec"] = round(time.time() - t0, 3)

        except Exception as e:
            record["status"] = "failed"
            record["error"] = f"{type(e).__name__}: {e}"
            record["elapsed_sec"] = round(time.time() - t0, 3)
            if not args.continue_on_error:
                append_manifest(Path(args.manifest), record)
                raise
        finally:
            append_manifest(manifest_path, record)
            summary["records"].append(record)

    summary["finished"] = datetime.now().isoformat()
    # also save an overall summary for convenience
    (manifest_path.parent / "batch_summary.json").write_text(json.dumps(summary, indent=2))
    print("✅ Batch complete.")
    print(f"📝 Manifest: {manifest_path}")
    print(f"🧾 Summary:  {(manifest_path.parent / 'batch_summary.json')}")
    

if __name__ == "__main__":
    main()