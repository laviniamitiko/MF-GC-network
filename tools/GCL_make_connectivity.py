#!/usr/bin/env python3
"""
tools/GCL_make_connectivity.py

Generate MF–GrC connectivity with fixed fan-in N_syn=4 per GrC.

Writes to:
  network_structures/N_grc_{N}_N_mf_{M}{suffix}.pkl
  network_structures/N_grc_{N}_N_mf_{M}{suffix}.json

Notes:
- Builds glomerulus→GrC first, then collapses to MF→GrC (binary):
  MF→GC = 1 iff ANY glomerulus on that MF connects to that GC.
- Ensures every GC column sums to 4.
- Exposes generate_connectivity(outdir, seed, **kwargs) for Python imports.
"""

from __future__ import annotations
import argparse, json, pickle as pkl
from pathlib import Path
from datetime import datetime
from typing import Optional, Sequence, Tuple, Dict, Any

import numpy as np


# ----------------- helpers (adapted from your script) -----------------
def generate_grc_positions(density: float, diam: float) -> np.ndarray:
    avg_num_incube = int(diam**3 * density)
    grc_pos = np.random.uniform(low=-diam/2, high=diam/2, size=(avg_num_incube, 3))
    in_ball = np.sqrt((grc_pos**2).sum(axis=1)) <= (diam/2)
    return grc_pos[in_ball, :]

def renumber(glom_mf_id: np.ndarray) -> np.ndarray:
    N_glom = glom_mf_id.shape[0]
    mfs = np.unique(glom_mf_id)
    glom_mf_id_new = np.zeros((N_glom,), int)
    for k, mf in enumerate(mfs):
        glom_mf_id_new[np.where(glom_mf_id == mf)[0]] = k
    return glom_mf_id_new

def draw_from_p_num_glom(p: np.ndarray, numsamples: int) -> np.ndarray:
    values = np.arange(1, len(p)+1.)
    p_cum = np.cumsum(np.append([0], p[0:-1]))
    samples = np.zeros((numsamples))
    randvars = np.random.uniform(0, 1, numsamples)
    for k in range(numsamples):
        samples[k] = values[np.where(randvars[k] > p_cum)[0][-1]]
    return samples

def generate_glom_positions(p_num_glom: np.ndarray, glom_density: float,
                            dx: float, dy: float, dz: float, diam: float) -> Tuple[np.ndarray, np.ndarray]:
    avg_glom_per_mf = (p_num_glom * np.arange(1, len(p_num_glom)+1)).sum()
    big_diam = diam * 3
    avg_total_glom_incube = big_diam**3 * glom_density
    avg_mf_incube = int(avg_total_glom_incube / avg_glom_per_mf)

    num_glom = draw_from_p_num_glom(p_num_glom, avg_mf_incube)
    glom_mf_id = np.zeros(int(num_glom.sum()), int)
    glom_pos = np.zeros((int(num_glom.sum()), 3))
    ix = 0
    for k in range(avg_mf_incube):
        glom_mf_id[ix:ix+int(num_glom[k])] = k
        glom_pos[ix, :] = np.random.uniform(low=-big_diam/2, high=big_diam/2, size=(3))
        for j in range(1, int(num_glom[k])):
            glom_pos[ix+j] = glom_pos[ix+j-1] + np.array([
                (-1)**np.round(np.random.uniform()) * np.random.exponential(scale=dx),
                (-1)**np.round(np.random.uniform()) * np.random.exponential(scale=dy),
                (-1)**np.round(np.random.uniform()) * np.random.exponential(scale=dz)
            ])
        ix += int(num_glom[k])

    in_ball = np.sqrt((glom_pos**2).sum(axis=1)) <= (diam/2)
    glom_pos = glom_pos[in_ball, :]
    glom_mf_id = renumber(glom_mf_id[in_ball])
    return glom_pos, glom_mf_id

def get_degreedist(glom_mf_id: np.ndarray, N_grc: int, N_glom: int, d: int) -> np.ndarray:
    ddist = np.zeros((N_glom), int)
    N_mf = np.unique(glom_mf_id).shape[0]
    for _ in range(N_grc):
        mf_chosen = np.random.choice(range(N_mf), size=d, replace=False)
        for mf in mf_chosen:
            gl = np.random.choice(np.where(glom_mf_id == mf)[0])
            ddist[gl] += 1
    return ddist

def closest_allowed_grc(attached_grcs: Sequence[int], grc_pos: np.ndarray, this_glom_pos: np.ndarray,
                        conn_mat: np.ndarray, d: int, dlen: float) -> int:
    N_grc = grc_pos.shape[0]
    grcs_not_yet_attached = [n for n in range(N_grc) if n not in attached_grcs]
    grcs_not_yet_full = [n for n in range(N_grc) if conn_mat[:, n].sum() < d]
    grcs_available = [n for n in grcs_not_yet_full if n in grcs_not_yet_attached]
    dists_from_glom = np.sqrt(((grc_pos - this_glom_pos)**2).sum(axis=1))
    dists = np.abs(dists_from_glom - dlen)
    if len(dists[grcs_available]) > 0:
        return grcs_available[int(np.argmin(dists[grcs_available]))]
    return -1

def check_valid_connectivity_glom(conn_mat: np.ndarray, ddist: np.ndarray,
                                  glom_mf_id: np.ndarray, d: int) -> None:
    N_mf = np.unique(glom_mf_id).shape[0]
    for mf in range(N_mf):
        gloms = np.where(glom_mf_id == mf)[0]
        assert np.all(conn_mat[gloms, :].sum(axis=0) <= 1), f"Mossy fiber {mf} has multiple gloms to the same GrC."
    assert np.all(conn_mat.sum(axis=1) - ddist == 0), "Glomerular degree distribution mismatch."
    assert np.all(conn_mat.sum(axis=0) == d), f"Not all GrCs have {d} dendrites."

def optswap(glom_incomplete: int, grcs_incomplete: Sequence[int], grcs_swappable: Sequence[int],
            grc_pos: np.ndarray, glom_pos: np.ndarray, conn_mat: np.ndarray,
            dlen: float, glom_mf_id: np.ndarray) -> np.ndarray:
    N_glom = glom_pos.shape[0]
    gloms_swappable = []
    for gi in grcs_incomplete:
        row = []
        for gs in grcs_swappable:
            row.append([gl for gl in range(N_glom)
                        if (conn_mat[gl, gs] == 1 and conn_mat[np.where(glom_mf_id == glom_mf_id[gl])[0], gi].sum() == 0)])
        gloms_swappable.append(row)

    deviation = np.zeros((len(grcs_incomplete), len(grcs_swappable)), float)
    index_ii = np.zeros_like(deviation, int)
    glom_i = glom_pos[glom_incomplete]
    for i, gi in enumerate(grcs_incomplete):
        grc_A = grc_pos[gi]
        for j, gs in enumerate(grcs_swappable):
            grc_B = grc_pos[gs]
            dist_iB = np.sqrt(((glom_i - grc_B)**2).sum())
            gloms = gloms_swappable[i][j]
            if len(gloms) == 0:
                deviation[i, j] = np.inf
                continue
            devs = np.zeros((len(gloms)), float)
            for k, gl_k in enumerate(gloms):
                dist_iiA = np.sqrt(((glom_pos[gl_k] - grc_A)**2).sum())
                devs[k] = (dlen - dist_iB)**2 + (dlen - dist_iiA)**2
            deviation[i, j] = devs.min()
            index_ii[i, j] = int(devs.argmin())

    iA, iB = np.unravel_index(np.nanargmin(deviation), deviation.shape)
    opt_grc_A = grcs_incomplete[iA]
    opt_grc_B = grcs_swappable[iB]
    opt_glom_ii = gloms_swappable[iA][iB][index_ii[iA, iB]]

    assert conn_mat[opt_glom_ii, opt_grc_B] == 1
    assert conn_mat[glom_incomplete, opt_grc_A] == 1
    assert conn_mat[opt_glom_ii, opt_grc_A] == 0

    conn_mat[opt_glom_ii, opt_grc_B] = 0
    conn_mat[opt_glom_ii, opt_grc_A] = 1
    conn_mat[glom_incomplete, opt_grc_B] = 1
    return conn_mat

def shuffle_conns(grc_pos: np.ndarray, glom_pos: np.ndarray, d: int, dlen: float,
                  gloms_incomplete: Sequence[int], conn_mat: np.ndarray,
                  ddist: np.ndarray, glom_mf_id: np.ndarray) -> np.ndarray:
    N_grc = grc_pos.shape[0]
    for gl in gloms_incomplete:
        incomplete_conns = int(ddist[gl] - conn_mat[gl, :].sum())
        for _ in range(incomplete_conns):
            grcs_incomplete = [n for n in range(N_grc) if conn_mat[:, n].sum() < d]
            grcs_swappable  = [n for n in range(N_grc) if (n not in grcs_incomplete and conn_mat[gl, n] == 0)]
            conn_mat = optswap(gl, grcs_incomplete, grcs_swappable, grc_pos, glom_pos, conn_mat, dlen, glom_mf_id)
    return conn_mat

def alg_connections(grc_pos: np.ndarray, glom_pos: np.ndarray, glom_mf_id: np.ndarray,
                    d: int, dlen: float) -> Tuple[np.ndarray, np.ndarray]:
    N_grc = grc_pos.shape[0]; N_glom = glom_pos.shape[0]
    ddist = get_degreedist(glom_mf_id, N_grc, N_glom, d)
    conn_mat = np.zeros((N_glom, N_grc), int)
    for conn in range(1, int(ddist.max())+1):
        for gl in range(N_glom):
            if conn <= ddist[gl]:
                mf = glom_mf_id[gl]
                glom_on_mf = np.where(glom_mf_id == mf)[0]
                attached_grcs = np.where(conn_mat[glom_on_mf].sum(axis=0))[0]
                grc = closest_allowed_grc(attached_grcs, grc_pos, glom_pos[gl, :], conn_mat, d, dlen)
                if grc >= 0:
                    conn_mat[gl, grc] = 1
    if not np.all(conn_mat.sum(axis=1) - ddist == 0):
        gloms_incomplete = np.where(conn_mat.sum(axis=1) != ddist)[0]
        conn_mat = shuffle_conns(grc_pos, glom_pos, d, dlen, gloms_incomplete, conn_mat, ddist, glom_mf_id)
    check_valid_connectivity_glom(conn_mat, ddist, glom_mf_id, d)
    return conn_mat, ddist


# ----------------- public API -----------------
def generate_connectivity(
    outdir: str | Path,
    seed: Optional[int] = None,
    *,
    glom_density: float = 6.6e-4,
    grc_density: float = 1.9e-3,
    glom_dx: float = 60.0,
    glom_dy: float = 20.0,
    glom_dz: float = 2.0,
    diam: float = 80.0,
    dlen: float = 15.0,
    p_num_glom: Sequence[float] = (0.0, 45.0, 17.0, 8.0, 5.0),
    suffix: str = "",
) -> str:
    """
    Build a connectivity and save it under `outdir`. Returns ABSOLUTE path to the .pkl.

    - N_syn (fanin) is fixed at 4 by design.
    - `suffix` lets you avoid overwriting files with identical (N_grc, N_mf).
      e.g., suffix="_seed123" or suffix="__v2".
    """
    if seed is not None:
        np.random.seed(seed)

    d = 4  # fixed fan-in
    p = np.array(p_num_glom, dtype=float); p = p / p.sum()

    grc_pos = generate_grc_positions(grc_density, diam)
    glom_pos, glom_mf_id = generate_glom_positions(p, glom_density, glom_dx, glom_dy, glom_dz, diam)

    N_glom = glom_pos.shape[0]; N_grc = grc_pos.shape[0]
    conn_glom_grc, ddist = alg_connections(grc_pos, glom_pos, glom_mf_id, d, dlen)

    mfs = np.unique(glom_mf_id); N_mf = mfs.size
    conn_mf_grc = np.zeros((N_mf, N_grc), dtype=np.uint8)
    for i_mf, mf in enumerate(mfs):
        gloms = np.where(glom_mf_id == mf)[0]
        conn_mf_grc[i_mf, :] = (conn_glom_grc[gloms, :].sum(axis=0) > 0).astype(np.uint8)

    # sanity: every GC has fan-in = 4
    fanin = conn_mf_grc.sum(axis=0)
    assert np.all(fanin == d), f"Expected fan-in 4; got {np.unique(fanin)}"

    outdir = Path(outdir).resolve(); outdir.mkdir(parents=True, exist_ok=True)
    stem = f"N_grc_{N_grc}_N_mf_{N_mf}{suffix}"
    pkl_path = outdir / f"{stem}.pkl"
    json_path = outdir / f"{stem}.json"

    payload: Dict[str, Any] = {
        "conn_mat": conn_mf_grc,            # MF × GC
        "conn_glom_grc": conn_glom_grc,     # glomerulus × GC
        "ddist_glom": ddist,
        "glom_pos": glom_pos,
        "grc_pos": grc_pos,
        "glom_mf_id": glom_mf_id,
        "params": {
            "seed": seed,
            "glom_density": glom_density,
            "grc_density":  grc_density,
            "glom_dx": glom_dx, "glom_dy": glom_dy, "glom_dz": glom_dz,
            "diam": diam, "dlen": dlen,
            "p_num_glom": list(p),
            "generated_at": datetime.now().isoformat()
        }
    }
    with open(pkl_path, "wb") as f:
        pkl.dump(payload, f)

    json_path.write_text(json.dumps({
        "pickle": str(pkl_path),
        "N_mf": int(N_mf),
        "N_glom": int(N_glom),
        "N_grc": int(N_grc),
        "fan_in": 4,
        "seed": seed
    }, indent=2))

    return str(pkl_path)


# ----------------- CLI -----------------
def _cli():
    ap = argparse.ArgumentParser(description="Generate MF→GrC connectivity (N_syn=4) and save to network_structures/")
    ap.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")

    # biological/geometric params (defaults from your code)
    ap.add_argument("--glom-density", type=float, default=6.6e-4)
    ap.add_argument("--grc-density",  type=float, default=1.9e-3)
    ap.add_argument("--glom-dx", type=float, default=60.0)
    ap.add_argument("--glom-dy", type=float, default=20.0)
    ap.add_argument("--glom-dz", type=float, default=2.0)
    ap.add_argument("--diam",    type=float, default=80.0)
    ap.add_argument("--dlen",    type=float, default=15.0)
    ap.add_argument("--p-num-glom", type=float, nargs="+",
                    default=[0.0, 45.0, 17.0, 8.0, 5.0],
                    help="Probabilities for a MF to have 1..k glomeruli (Sultan et al.)")

    # outputs
    default_outdir = Path(__file__).resolve().parents[1] / "network_structures"
    ap.add_argument("--outdir", default=str(default_outdir))
    ap.add_argument("--suffix", default="", help="Append to filename to prevent overwrite (e.g. _seed123)")

    args = ap.parse_args()

    path = generate_connectivity(
        outdir=args.outdir,
        seed=args.seed,
        glom_density=args.glom_density,
        grc_density=args.grc_density,
        glom_dx=args.glom_dx, glom_dy=args.glom_dy, glom_dz=args.glom_dz,
        diam=args.diam, dlen=args.dlen,
        p_num_glom=args.p_num_glom,
        suffix=args.suffix,
    )
    # Print path so orchestrators (or you) can capture it
    print(path)

if __name__ == "__main__":
    _cli()
