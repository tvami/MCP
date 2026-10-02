#!/usr/bin/env python3
"""
Cross section vs mass (one curve per charge) and vs charge (one curve per mass)
for the HSCP-MCP MadGraph gridpacks on EOS.

The cross section is the MG integration result ("pilotrun") in each tarball's
gridpack_generation.log, streamed over xrootd. Results are cached in a CSV so
the plotting can be redone without re-reading EOS.

Usage (needs a valid grid proxy and matplotlib, e.g. a CMSSW env):
  python3 plot_gridpack_xsec.py --eos-dir /eos/user/t/tvami/output/13p6TeVPhotonFusion \
      --energy "13.6 TeV" --process "photon fusion" --out-dir xsec_13p6TeVPhotonFusion
"""

import argparse
import csv
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REDIRECTOR = "root://eosuser.cern.ch"
NAME_RE = re.compile(r"Par-(S-0-)?M-(\d+)-Q-(\d+)_")
XSEC_RE = re.compile(r"Results Summary for run: pilotrun.*?Cross-section :\s+([0-9.eE+-]+) \+- ([0-9.eE+-]+) pb", re.S)


def list_tarballs(eos_dir):
    out = subprocess.run(["xrdfs", REDIRECTOR, "ls", eos_dir], capture_output=True, text=True, check=True).stdout
    return sorted(l for l in out.split() if l.endswith("_tarball.tar.xz"))


def read_xsec(path):
    m = NAME_RE.search(os.path.basename(path))
    spin = "0" if m.group(1) else "1/2"
    mass, charge = int(m.group(2)), int(m.group(3))  # charge in units of e/3
    # stream the tarball, keep only the log
    cmd = f"xrdcp -s {REDIRECTOR}/{path} - | tar -xJO gridpack_generation.log"
    log = subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout
    x = XSEC_RE.search(log)
    if not x:
        print(f"WARNING: no pilotrun cross section in {os.path.basename(path)}")
        return dict(spin=spin, mass=mass, charge=charge, xsec_pb="", err_pb="")
    return dict(spin=spin, mass=mass, charge=charge, xsec_pb=float(x.group(1)), err_pb=float(x.group(2)))


def extract(eos_dir, csv_path, workers):
    files = list_tarballs(eos_dir)
    print(f"Reading {len(files)} gridpacks from {eos_dir} with {workers} workers")
    with ThreadPoolExecutor(workers) as ex:
        rows = list(ex.map(read_xsec, files))
    rows.sort(key=lambda r: (r["spin"], r["charge"], r["mass"]))
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["spin", "mass", "charge", "xsec_pb", "err_pb"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {csv_path}")


def load(csv_path):
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            if r["xsec_pb"] == "":
                continue
            rows.append((r["spin"], int(r["mass"]), int(r["charge"]), float(r["xsec_pb"]), float(r["err_pb"])))
    return rows


def qlabel(c):
    # charge stored in units of e/3; integer e when divisible
    return f"Q = {c // 3}e" if c % 3 == 0 else f"Q = {c}/3 e"


def plot(rows, spin, energy, process, out_dir):
    sel = [r for r in rows if r[0] == spin]
    if not sel:
        return
    charges = sorted({r[2] for r in sel})
    masses = sorted({r[1] for r in sel})
    tag = "Spin0" if spin == "0" else "SpinHalf"
    title_spin = f"spin-{spin}"

    # sigma vs mass, one curve per charge
    fig, ax = plt.subplots(figsize=(11, 7.5))
    colors = plt.cm.turbo(np.linspace(0.0, 1.0, len(charges)))
    for c, col in zip(charges, colors):
        pts = sorted((r[1], r[3], r[4]) for r in sel if r[2] == c)
        m, x, e = map(np.array, zip(*pts))
        ax.errorbar(m, x, yerr=e, marker="o", ms=5, capsize=3, color=col, label=qlabel(c))
    ax.set_yscale("log")
    ax.set_xlabel("Mass [GeV]", fontsize=14)
    ax.set_ylabel(r"$\sigma$ [pb]", fontsize=14)
    ax.set_title(f"HSCP-MCP {process} pair production cross section ({title_spin}) @ {energy}", fontsize=14)
    ax.grid(True, ls="--", alpha=0.5)
    ax.legend(title="Charge", ncol=2, fontsize=9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"xsec_vs_mass_{tag}.{ext}"))
    plt.close(fig)

    # sigma vs charge, one curve per mass
    fig, ax = plt.subplots(figsize=(11, 7.5))
    colors = plt.cm.viridis(np.linspace(0.0, 0.95, len(masses)))
    for mval, col in zip(masses, colors):
        pts = sorted((r[2] / 3.0, r[3], r[4]) for r in sel if r[1] == mval)
        q, x, e = map(np.array, zip(*pts))
        ax.errorbar(q, x, yerr=e, marker="o", ms=5, capsize=3, color=col, label=f"M = {mval} GeV")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Charge |Q| [e]", fontsize=14)
    ax.set_ylabel(r"$\sigma$ [pb]", fontsize=14)
    ax.set_title(f"HSCP-MCP {process} pair production cross section ({title_spin}) @ {energy}", fontsize=14)
    ax.grid(True, which="both", ls="--", alpha=0.4)
    ax.legend(title="Mass", ncol=2, fontsize=9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"xsec_vs_charge_{tag}.{ext}"))
    plt.close(fig)
    print(f"Plotted {title_spin}: {len(charges)} charges x {len(masses)} masses")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eos-dir", required=True)
    p.add_argument("--energy", default="13.6 TeV")
    p.add_argument("--process", default="photon fusion")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--workers", type=int, default=8)  # uaf caps users at 8 cores
    p.add_argument("--plot-only", action="store_true", help="reuse the cached CSV")
    a = p.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    csv_path = os.path.join(a.out_dir, "xsec.csv")
    if not a.plot_only:
        extract(a.eos_dir, csv_path, a.workers)
    rows = load(csv_path)
    for spin in ("1/2", "0"):
        plot(rows, spin, a.energy, a.process, a.out_dir)


if __name__ == "__main__":
    main()
