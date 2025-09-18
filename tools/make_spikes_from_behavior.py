#!/usr/bin/env python3
"""
Generate behavior-modulated MF SpikeArrays (NeuroML) with controllable PM/NM fractions.

Auto-naming: if --out is NOT given, the output file is:
  spikes/<beh_stem>__<conn_stem>__<connhash12>__pm<pm>_nm<nm>__t<minT>-<maxT>.nml

Example:
  python tools/make_spikes_from_behavior.py \
    --connectivity network_structures/N_grc_486_N_mf_182.pkl \
    --beh-mat data/191018_13_39_41_dfTrace.mat \
    --minT 100 --maxT 140 --acq-rate 7.2 \
    --pm-frac 0.40 --nm-frac 0.30
"""

from __future__ import annotations
import argparse, json, pickle as pkl, re, hashlib
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from scipy.io import loadmat
import neuroml as nml
from neuroml import NeuroMLDocument, SpikeArray, Spike
from pyneuroml import pynml


# ---------- small helpers ----------
def sanitize(stem: str) -> str:
    """Keep it filesystem-friendly."""
    s = stem.strip()
    s = re.sub(r"[^\w\-.]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "beh"

def conn_hash12(conn_mat: np.ndarray) -> str:
    return hashlib.sha256(conn_mat.astype(np.uint8).tobytes()).hexdigest()[:12]

# ---------- behavior → rates ----------
def load_behavior_vector(beh_mat_path: Path) -> np.ndarray:
    m = loadmat(str(beh_mat_path))
    # NB: you said the variable is state_ds in your notes earlier; adjust here if needed
    if "state_ds" not in m:
        raise ValueError(f"'{beh_mat_path}': variable 'state_ds' not found")
    v = m["state_ds"]
    if v.ndim != 2:
        raise ValueError(f"'state_ds' must be 2D, got shape {v.shape}")
    if v.shape[0] == 1 and v.shape[1] >= 1:
        v = v.T
    elif v.shape[1] != 1:
        raise ValueError(f"'state_ds' expected (T,1) or (1,T); got {v.shape}")
    return np.asarray(v, dtype=float).reshape(-1, 1)  # (T,1)

def generate_rates_from_behavior(
    beh_vec: np.ndarray,         # (T,1)
    minT: float,
    maxT: float,
    n_mf: int,
    acquisition_rate: float,     # Hz
    scale: float,
    offset: float,
    minrate: float,
    sparsity: float = 0.0,
    weights: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if seed is not None:
        np.random.seed(seed)

    T_total = beh_vec.shape[0]
    min_idx = int(minT * acquisition_rate)
    max_idx = int(maxT * acquisition_rate)
    if min_idx < 0 or max_idx > T_total or min_idx >= max_idx:
        total_s = T_total / acquisition_rate
        raise ValueError(f"Invalid window [{minT},{maxT}] vs data length {total_s:.3f}s")

    beh = beh_vec[min_idx:max_idx, :]  # (T,1)
    time = np.arange(min_idx, max_idx) / acquisition_rate  # seconds

    if weights is None:
        weights = np.random.rand(n_mf, 1)
        mask = np.random.rand(n_mf, 1) < sparsity
        weights[mask] = 0.0
    else:
        assert weights.shape == (n_mf, 1)

    rates = (beh @ weights.T) * scale + offset
    rates = np.maximum(rates, minrate)
    return time, rates, weights

# ---------- inhomogeneous Poisson via thinning ----------
def getrate(rate: np.ndarray, x: np.ndarray, rate_dt: float) -> np.ndarray:
    rate = np.asarray(rate).flatten()
    idx = np.floor(np.array(x) / rate_dt).astype(int)
    idx = np.clip(idx, 0, len(rate) - 1)
    return rate[idx]

def create_spike_array_from_rate(
    rate: np.ndarray,  # sampled at rate_dt
    duration: float,   # seconds (simulation duration)
    spike_dt: float,   # seconds (sampling grid for comparison only)
    rate_dt: float,    # seconds (1/acquisition_rate)
    name: str,
    seed: Optional[int] = None,
) -> SpikeArray:
    if seed is not None:
        np.random.seed(seed)
    m2 = float(np.max(rate))
    sa = SpikeArray(id=name)
    if m2 <= 0:
        return sa
    n_spikes = int(np.ceil(1.5 * duration * m2))
    u = np.random.random((n_spikes, 1))
    y = np.cumsum(-np.log(u) / m2).flatten()
    y = y[y <= duration]
    keep = np.random.rand(len(y)) < (getrate(rate, y, rate_dt) / m2)
    for j, t in enumerate(y[keep]):
        sa.spikes.append(Spike(id=j, time=f"{1e3 * t:.3f} ms"))  # ms
    return sa


# ---------- library entry point ----------
def make_spikes_from_behavior(
    connectivity: Path | str,
    beh_mat: Path | str,
    *,
    minT: float,
    maxT: float,
    acq_rate: float,
    spike_dt: float = 0.001,
    pm_frac: float,
    nm_frac: float,
    label: str = "states_2",
    out: Path | str | None = None,
    # PM/NM/NS rate params
    pm_scale: float = 50.0, pm_offset: float = 7.0,  pm_minrate: float = 2.0,
    nm_scale: float = -50.0, nm_offset: float = 40.0, nm_minrate: float = 2.0,
    ns_scale: float = 0.0, ns_offset: float = 7.0,  ns_minrate: float = 2.0,
    # seeds (use fixed values or derive outside)
    seed_pm: int = 1, seed_nm: int = 2, seed_ns: int = 3,
) -> Tuple[Path, Path]:
    """Build MF SpikeArrays + metadata json. Returns (nml_path, json_path)."""

    conn_path = Path(connectivity).resolve()
    with open(conn_path, "rb") as f:
        payload = pkl.load(f, encoding="latin1")
    if "conn_mat" not in payload:
        raise ValueError(f"{conn_path} missing 'conn_mat'")
    conn_mat = np.asarray(payload["conn_mat"])
    N_mf = int(conn_mat.shape[0])
    chash = conn_hash12(conn_mat)

    # fractions → counts
    if pm_frac < 0 or nm_frac < 0 or pm_frac + nm_frac > 1.0:
        raise ValueError("Require pm_frac >=0, nm_frac >=0, pm_frac+nm_frac <= 1.")
    n_pm = int(round(pm_frac * N_mf))
    n_nm = int(round(nm_frac * N_mf))
    if n_pm + n_nm > N_mf:
        n_nm = max(0, N_mf - n_pm)
    n_ns = N_mf - (n_pm + n_nm)

    beh_path = Path(beh_mat).resolve()
    beh_vec = load_behavior_vector(beh_path)

    rate_dt = 1.0 / float(acq_rate)
    duration = float(maxT - minT)

    spike_arrays_all: list[SpikeArray] = []

    if n_pm > 0:
        _, rates_pm, _ = generate_rates_from_behavior(
            beh_vec, minT, maxT, n_pm, acq_rate,
            scale=pm_scale, offset=pm_offset, minrate=pm_minrate,
            sparsity=0.0, seed=seed_pm,
        )
        for i in range(n_pm):
            spike_arrays_all.append(
                create_spike_array_from_rate(
                    rates_pm[:, i], duration, spike_dt, rate_dt, name=f"MF_PM_{i}",
                    seed=seed_pm + i
                )
            )

    if n_nm > 0:
        _, rates_nm, _ = generate_rates_from_behavior(
            beh_vec, minT, maxT, n_nm, acq_rate,
            scale=nm_scale, offset=nm_offset, minrate=nm_minrate,
            sparsity=0.0, seed=seed_nm,
        )
        for i in range(n_nm):
            spike_arrays_all.append(
                create_spike_array_from_rate(
                    rates_nm[:, i], duration, spike_dt, rate_dt, name=f"MF_NM_{i}",
                    seed=seed_nm + i
                )
            )

    if n_ns > 0:
        _, rates_ns, _ = generate_rates_from_behavior(
            beh_vec, minT, maxT, n_ns, acq_rate,
            scale=ns_scale, offset=ns_offset, minrate=ns_minrate,
            sparsity=0.0, seed=seed_ns,
        )
        for i in range(n_ns):
            spike_arrays_all.append(
                create_spike_array_from_rate(
                    rates_ns[:, i], duration, spike_dt, rate_dt, name=f"MF_NS_{i}",
                    seed=seed_ns + i
                )
            )

    assert len(spike_arrays_all) == N_mf, f"Expected {N_mf} SpikeArrays, got {len(spike_arrays_all)}"

    # -------- auto-naming --------
    if out:
        out_path = Path(out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        beh_stem = sanitize(beh_path.stem)
        conn_stem = sanitize(conn_path.stem)
        pm = f"{pm_frac:.2f}".rstrip("0").rstrip(".")
        nm = f"{nm_frac:.2f}".rstrip("0").rstrip(".")
        tmin = f"{minT:g}"
        tmax = f"{maxT:g}"
        out_dir = Path("spikes").resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{beh_stem}__{conn_stem}__{chash}__pm{pm}_nm{nm}__t{tmin}-{tmax}.nml"

    # write NeuroML + metadata
    doc = NeuroMLDocument(id="MF_StatesOnly2")
    for sa in spike_arrays_all:
        doc.spike_arrays.append(sa)
    pynml.write_neuroml2_file(doc, str(out_path))
    meta_path = out_path.with_suffix(".json")

    meta = {
        "connectivity": str(conn_path),
        "connectivity_hash12": chash,
        "behavior_mat": str(beh_path),
        "N_mf": N_mf,
        "pm_frac": pm_frac,
        "nm_frac": nm_frac,
        "ns_frac": 1.0 - (pm_frac + nm_frac),
        "counts": {"PM": n_pm, "NM": n_nm, "NS": n_ns},
        "minT": minT,
        "maxT": maxT,
        "acq_rate": acq_rate,
        "spike_dt": spike_dt,
        "label": label,
        "seeds": {"PM": seed_pm, "NM": seed_nm, "NS": seed_ns},
        "rates": {
            "PM": {"scale": pm_scale, "offset": pm_offset, "minrate": pm_minrate},
            "NM": {"scale": nm_scale, "offset": nm_offset, "minrate": nm_minrate},
            "NS": {"scale": ns_scale, "offset": ns_offset, "minrate": ns_minrate},
        },
        "output_nml": str(out_path),
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    return out_path, meta_path


# ---------- CLI ----------
def _cli():
    ap = argparse.ArgumentParser(description="Make behavior-modulated MF SpikeArrays (NeuroML) with PM/NM fractions.")
    ap.add_argument("--connectivity", required=True,
                    help="Connectivity .pkl to read N_mf (MF×GC 'conn_mat').")
    ap.add_argument("--beh-mat", required=True,
                    help="Path to .mat containing 'state_ds' (T×1 or 1×T).")
    ap.add_argument("--minT", type=float, required=True)
    ap.add_argument("--maxT", type=float, required=True)
    ap.add_argument("--acq-rate", type=float, required=True)
    ap.add_argument("--spike-dt", type=float, default=0.001)
    ap.add_argument("--pm-frac", type=float, required=True)
    ap.add_argument("--nm-frac", type=float, required=True)
    ap.add_argument("--label", default="states_2")
    ap.add_argument("--out", default=None, help="Explicit output .nml path. If omitted, auto-named in spikes/.")
    # model params
    ap.add_argument("--pm-scale", type=float, default=50.0)
    ap.add_argument("--pm-offset", type=float, default=7.0)
    ap.add_argument("--pm-minrate", type=float, default=2.0)
    ap.add_argument("--nm-scale", type=float, default=-50.0)
    ap.add_argument("--nm-offset", type=float, default=40.0)
    ap.add_argument("--nm-minrate", type=float, default=2.0)
    ap.add_argument("--ns-scale", type=float, default=0.0)
    ap.add_argument("--ns-offset", type=float, default=7.0)
    ap.add_argument("--ns-minrate", type=float, default=2.0)
    # seeds
    ap.add_argument("--seed-pm", type=int, default=1)
    ap.add_argument("--seed-nm", type=int, default=2)
    ap.add_argument("--seed-ns", type=int, default=3)
    args = ap.parse_args()

    nml_path, meta_path = make_spikes_from_behavior(
        connectivity=args.connectivity,
        beh_mat=args.beh_mat,
        minT=args.minT,
        maxT=args.maxT,
        acq_rate=args.acq_rate,
        spike_dt=args.spike_dt,
        pm_frac=args.pm_frac,
        nm_frac=args.nm_frac,
        label=args.label,
        out=args.out,
        pm_scale=args.pm_scale, pm_offset=args.pm_offset, pm_minrate=args.pm_minrate,
        nm_scale=args.nm_scale, nm_offset=args.nm_offset, nm_minrate=args.nm_minrate,
        ns_scale=args.ns_scale, ns_offset=args.ns_offset, ns_minrate=args.ns_minrate,
        seed_pm=args.seed_pm, seed_nm=args.seed_nm, seed_ns=args.seed_ns,
    )
    print(f"✅ Saved MF SpikeArrays to: {nml_path}")
    print(f"📝 Wrote metadata to:       {meta_path}")

if __name__ == "__main__":
    _cli()
