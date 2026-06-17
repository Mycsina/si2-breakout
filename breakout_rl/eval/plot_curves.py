"""Render training curves from one or more run log.csv files into PNGs for the report."""
import argparse, glob, csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load(path):
    """Read (x, y) eval points from a run log. Auto-detects the schema: the flat DQN
    logs step/eval_score; the SMDP agent logs option/eval_clears. Returns the y-column
    name too so the caller can label the axis correctly."""
    steps, evals = [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        xcol = "step" if "step" in fields else "option"
        ycol = "eval_score" if "eval_score" in fields else "eval_clears"
        for row in reader:
            if row.get(ycol):
                steps.append(int(row[xcol])); evals.append(float(row[ycol]))
    return steps, evals, ycol


def main(pattern, out):
    plt.figure(); ylabel = "eval score"
    for path in glob.glob(pattern):
        s, e, ycol = load(path)
        if s:
            plt.plot(s, e, label=path.split("/")[-2])
            ylabel = "eval score" if ycol == "eval_score" else "eval clears (boards cleared)"
    plt.xlabel("training step"); plt.ylabel(ylabel); plt.legend(); plt.grid(True)
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pattern", default="checkpoints/flat_dqn*/log.csv")
    p.add_argument("--out", default="docs/curves_flat_dqn.png")
    a = p.parse_args(); main(a.pattern, a.out)
