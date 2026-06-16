"""Render training curves from one or more run log.csv files into PNGs for the report."""
import argparse, glob, csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load(path):
    steps, evals = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            if row.get("eval_score"):
                steps.append(int(row["step"])); evals.append(float(row["eval_score"]))
    return steps, evals


def main(pattern, out):
    plt.figure()
    for path in glob.glob(pattern):
        s, e = load(path)
        if s:
            plt.plot(s, e, label=path.split("/")[-2])
    plt.xlabel("training step"); plt.ylabel("eval score"); plt.legend(); plt.grid(True)
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pattern", default="checkpoints/flat_dqn*/log.csv")
    p.add_argument("--out", default="docs/curves_flat_dqn.png")
    a = p.parse_args(); main(a.pattern, a.out)
