#!/usr/bin/env python3
import subprocess, sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE / "results" / "behmod" / "conn"

def main():
    lems_files = sorted(ROOT.rglob("LEMS*.xml"))
    if not lems_files:
        print("No LEMS*.xml under results/behmod/conn/ — did you run --emit-only?")
        return 1

    run_dirs = sorted({lf.parent for lf in lems_files})
    print(f"Found {len(run_dirs)} run dirs")

    for rd in run_dirs:
        candidates = sorted(rd.glob("LEMS_Sim*.xml")) or sorted(rd.glob("LEMS*.xml"))
        lems = candidates[0]
        print(f"[prep] {rd}: {lems.name}")
        subprocess.check_call(
            ["pynml", lems.name, "-neuron", "-nogui"],
            cwd=rd
        )
    print("Done.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
